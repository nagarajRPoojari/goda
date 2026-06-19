import torch
import triton  #  type: ignore pyright: ignore[import] ofcourse triton is not there in mac
import triton.language as tl  # type: ignore pyright: ignore[import] ofcourse triton is not there in mac


@triton.jit
def _attn_backward_preprocess_D(
    O, dO, D, SEQ_LEN, BLOCK_SIZE_Q: tl.constexpr, HEAD_DIM: tl.constexpr
):

    index_O_block = tl.program_id(0)
    index_O_batch_head = tl.program_id(1)

    # [x, x+1, x+2, .. , x+BLOCK_SIZE_Q]
    offset_O_T = index_O_block * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # [0, 1, ... , HEAD_DIM-1]
    offset_O_D = tl.arange(0, HEAD_DIM)

    block_O = tl.load(
        O
        + index_O_batch_head * HEAD_DIM * SEQ_LEN
        + offset_O_T[:, None] * HEAD_DIM
        + offset_O_D[None, :]
    )

    block_dO = tl.load(
        dO
        + index_O_batch_head * HEAD_DIM * SEQ_LEN
        + offset_O_T[:, None] * HEAD_DIM
        + offset_O_D[None, :]
    )

    block_D = tl.sum(block_dO * block_O, 1)

    block_D_ptr = D + index_O_batch_head * SEQ_LEN + offset_O_T
    tl.store(block_D_ptr, block_D)


@triton.jit
def _attn_backward_dK_dV(
    Q,
    K,
    V,
    tau,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):

    index_batch_head = tl.program_id(2)
    index_KV_block = tl.program_id(0)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    offset_B_H = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offset_B_H_T = (index_batch_head * SEQ_LEN).to(tl.int64)

    # skip blocks and heads
    Q += offset_B_H
    K += offset_B_H
    V += offset_B_H
    dO += offset_B_H
    dQ += offset_B_H
    dK += offset_B_H
    dV += offset_B_H

    M += offset_B_H_T
    D += offset_B_H_T

    offset_D = tl.arange(0, HEAD_DIM)

    start_kv = index_KV_block * BLOCK_SIZE_KV
    offset_KV_T = start_kv + tl.arange(0, BLOCK_SIZE_KV)

    # output blocks dK, dV in SMEM
    block_dV = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
    block_dK = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)

    # fix K,V and loop over Qi
    # This works here since we are not longer restricted by softmax computation
    # we will simply reuse the softmax/logSumExp computed in fwd pass
    block_K = tl.load(  # [BLOCK_SIZE_KV, HEAD_DIM]
        K + offset_KV_T[:, None] * stride_seq + offset_D[None, :] * stride_dim
    )
    block_V = tl.load(  # [BLOCK_SIZE_KV, HEAD_DIM]
        V + offset_KV_T[:, None] * stride_seq + offset_D[None, :] * stride_dim
    )
    offset_Q_T = tl.arange(0, BLOCK_SIZE_Q)

    # note: i am gonna load Q as transposed to avoid further unnessary transpose ops
    block_Qt_ptr = Q + offset_Q_T[None, :] * stride_seq + offset_D[:, None] * stride_dim
    block_dO_ptr = dO + offset_Q_T[:, None] * stride_seq + offset_D[None, :] * stride_dim

    curr_offset_Q_T = 0
    for q_i in range(SEQ_LEN // BLOCK_SIZE_Q):
        # Update offset BEFORE loading
        offset_Q_T = curr_offset_Q_T + tl.arange(0, BLOCK_SIZE_Q)

        block_Qt = tl.load(block_Qt_ptr)
        # we dont need one more offset_D since shape of M is [BATCH_SIZE, NUM_HEADS, SEQ_LEN] only
        m_i = tl.load(M + offset_Q_T)
        block_dO = tl.load(block_dO_ptr)
        # delta D = rowsum(dO * O)
        block_D = tl.load(D + offset_Q_T)

        # we need (QK^T)^T = (K^T)^T (Q^T) = K(Q^T)
        block_QKt = tau * tl.dot(block_K, block_Qt)
        block_Pt = tl.math.exp(block_QKt - m_i[None, :])

        if STAGE == 3:
            # causal mask
            mask = offset_Q_T[None, :] >= offset_KV_T[:, None]
            block_Pt = tl.where(mask, block_Pt, 0.0)

        # accumulate dV
        # from FA1 dV <- dV + P^T dO
        block_dV += tl.dot(block_Pt.to(tl.float16), block_dO)

        # dP = dO * V^T
        # dP^T = V * dO^T
        block_dPt = tl.dot(block_V, tl.trans(block_dO)).to(tl.float32)

        # dS = P * (dP - D)  note: * is hadamard product here
        # => dS^T = P^T * (dP^T - D^T)
        block_dSt = block_Pt * (block_dPt - block_D[None, :])
        block_dSt = block_dSt.to(tl.float16)

        # dK <= dK + dS^T * Q
        block_dK += tau * tl.dot(block_dSt, tl.trans(block_Qt))

        curr_offset_Q_T += BLOCK_SIZE_Q
        block_Qt_ptr += BLOCK_SIZE_Q * stride_seq
        block_dO_ptr += BLOCK_SIZE_Q * stride_seq

    # SMEM to GMEM after full accumulation
    block_dV_ptr = dV + offset_KV_T[:, None] * stride_seq + offset_D[None, :] * stride_dim
    tl.store(block_dV_ptr, block_dV)

    block_dK_ptr = dK + offset_KV_T[:, None] * stride_seq + offset_D[None, :] * stride_dim
    tl.store(block_dK_ptr, block_dK)


@triton.jit
def _attn_backward_dQ(
    Q,
    K,
    V,
    tau,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):

    index_KV_block = tl.program_id(0)
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_B_H = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offset_B_H_T = (index_batch_head * SEQ_LEN).to(tl.int64)

    # skip blocks and heads
    Q += offset_B_H
    K += offset_B_H
    V += offset_B_H
    dO += offset_B_H
    dQ += offset_B_H
    dK += offset_B_H
    dV += offset_B_H

    M += offset_B_H_T
    D += offset_B_H_T

    # load scales
    offset_D = tl.arange(0, HEAD_DIM)

    start_q = index_KV_block * BLOCK_SIZE_Q
    offset_Q_T = start_q + tl.arange(0, BLOCK_SIZE_Q)

    block_Q = tl.load(Q + offset_Q_T[:, None] * stride_seq + offset_D[None, :] * stride_dim)
    block_dQ = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    block_dO = tl.load(dO + offset_Q_T[:, None] * stride_seq + offset_D[None, :] * stride_dim)

    block_M = tl.load(M + offset_Q_T)
    block_M = block_M[:, None]

    offset_KV_T = tl.arange(0, BLOCK_SIZE_KV)

    # We access the K and V as transposed blocks
    block_Kt_ptr = K + offset_KV_T[None, :] * stride_seq + offset_D[:, None] * stride_dim
    block_Vt_ptr = V + offset_KV_T[None, :] * stride_seq + offset_D[:, None] * stride_dim

    Di = tl.load(D + offset_Q_T)

    curr_offset_KV_T = 0
    for kv_i in range(SEQ_LEN // BLOCK_SIZE_KV):
        block_Kt = tl.load(block_Kt_ptr)
        block_Vt = tl.load(block_Vt_ptr)

        block_QK = tau * tl.dot(block_Q, block_Kt)
        block_P = tl.math.exp(block_QK - block_M)

        if STAGE == 3:
            offset_KV_T = curr_offset_KV_T + tl.arange(0, BLOCK_SIZE_KV)
            mask = offset_Q_T[:, None] >= offset_KV_T[None, :]
            block_P = tl.where(mask, block_P, 0.0)

        # dP and dS.
        block_dP = tl.dot(block_dO, block_Vt).to(tl.float32)
        block_dS = block_P * (block_dP - Di[:, None])
        block_dS = block_dS.to(tl.float16)

        # note: We need to de-scale dq in the end, because kT was pre-scaled.
        block_dQ += tau * tl.dot(block_dS, tl.trans(block_Kt))

        curr_offset_KV_T += BLOCK_SIZE_KV
        block_Kt_ptr += BLOCK_SIZE_KV * stride_seq
        block_Vt_ptr += BLOCK_SIZE_KV * stride_seq

    block_dQ_ptr = dQ + offset_Q_T[:, None] * stride_seq + offset_D[None, :] * stride_dim
    tl.store(block_dQ_ptr, block_dQ)


@triton.jit
def _attn_forward_internal(
    block_O,
    l_i,
    m_i,
    block_Q,
    block_K_ptr,
    block_V_ptr,
    index_Q_block,
    tau,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offset_Q_T: tl.constexpr,
    offset_KV_T: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # handling attention calculation in threee different stages
    # STAGE 1: lower left triangle matrix
    # STAGE 2: diagonal blocks
    # STAGE 3: uppoer right triangle matrix
    #
    # note: though it is confusing that range must be across cols, [None, None, .., lo:hi]
    # range is actually determined by Q block since we keep Q_block fix and loop over KV blocks
    if STAGE == 1:
        lo, hi = 0, index_Q_block * BLOCK_SIZE_Q
    elif STAGE == 2:
        lo, hi = index_Q_block * BLOCK_SIZE_Q, (index_Q_block + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(
            lo, BLOCK_SIZE_Q
        )  # seems to be hack that allows triton to optimize better : TODO: figure out how ?
    else:
        # upper right corner
        # only applicable for non-causal
        # though ideally it should be
        # lo, hi = (index_Q_block + 1) & BLOCK_SIZE_Q, SEQ_LEN
        # STAGE 3 is only applicable for non-causal and it needs full matrix attention
        # so, lo, hi = 0, SEQ_LEN
        # hoping we will not have any weird attention only future tokens (top right matrix)
        lo, hi = 0, SEQ_LEN

    # it's very confusing why we are moving by lo (BLOCK_SIZE_Q step size)
    # but it works
    block_K_ptr = tl.advance(block_K_ptr, (0, lo))
    block_V_ptr = tl.advance(block_V_ptr, (lo, 0))

    # fix Qi, loop over all K,V
    # Though it seems efficient to fix K,V and loop over all Q, reducing multipke K,V loads
    # e.g, 10 Q blocks 10 K, V blocks
    #       1. fix Q and loop over all K, V
    #          10 * (1 + 10 + 10) = 210 HBM loads
    #       2. fix K, V and loop over Q
    #          10 * (1 + 1 + 10)= 120 HBM loads
    # in forward pass we are dependent on prev K, V cols for softmax calculation across row.
    # i.e, Attn(Qi, Kj, Vj) depends on Attn(Qi, Kj-1, Vj-1)

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # GMEM to SMEM
        block_K = tl.load(block_K_ptr)
        block_V = tl.load(block_V_ptr)

        block_QK_T = tl.dot(block_Q, block_K)  # block K already transposed

        # STAGE 2 means we are at transition diagonal, be careful to apply mask
        if STAGE == 2:
            # offset_Q_T                     : [12, 13, 14, ... 12 + BLOCK_SIZE_Q]
            # offset_Q_T[:, None]            : [[12], [13], [14], .. [12 + BLOCK_SIZE_Q]]
            # offset_KV_T                    : [10, 11, 12, ... 10 + BLOCK_SIZE_KV]
            # offset_KV_T[None, :]           : [[10, 11, 12, .. 10 + BLOCK_SIZE_KV]]
            mask = offset_Q_T[:, None] >= (start_kv + offset_KV_T[None, :])
            block_QK_T = block_QK_T * tau + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(block_QK_T, 1))  # row max
            block_QK_T -= m_ij[:, None]

        else:
            m_ij = tl.maximum(m_i, tl.max(block_QK_T, 1) * tau)
            block_QK_T = block_QK_T * tau - m_ij[:, None]

        # exp(qk_ij - m_ij): [BLOCK_SIZE_Q, BLOCK_SIZE_KV]
        block_P = tl.math.exp(block_QK_T)

        # row sum (exp(qk_ij - m_ij)): [BLOCK_SIZE_KV]
        l_ij = tl.sum(block_P, 1)

        # correction factor
        alpha = tl.math.exp(m_i - m_ij)

        block_P = block_P.to(tl.float16)

        # l_i = l_i * exp(m_i - m_ij) + l_ij
        l_i = l_i * alpha + l_ij
        # O = O_old * alpha + PV
        block_O = block_O * alpha[:, None]
        block_O = tl.dot(block_P, block_V, block_O)

        m_i = m_ij

        block_K_ptr = tl.advance(block_K_ptr, (0, BLOCK_SIZE_KV))
        block_V_ptr = tl.advance(block_V_ptr, (BLOCK_SIZE_KV, 0))

    return block_O, l_i, m_i


# stride of ith dimension: how much step i need to take to move from j to j+1 in ith dim
# it is simply product of its inner dims
@triton.jit
def _attn_forward_kernel(
    Q,  # [B, H, T, D]
    K,  # [B, H, T, D]
    V,  # [B, H, T, D]
    tau,  # softmax scale , usuall 1/sqrt(D)
    M,  # [B, H, T]
    O,  # [B, H, T, D]
    stride_Q_B,
    stride_Q_H,
    stride_Q_T,
    stride_Q_D,
    stride_K_B,
    stride_K_H,
    stride_K_T,
    stride_K_D,
    stride_V_B,
    stride_V_H,
    stride_V_T,
    stride_V_D,
    stride_O_B,
    stride_O_H,
    stride_O_T,
    stride_O_D,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):

    index_Q_block = tl.program_id(0)
    index_Q_batch_head = tl.program_id(1)
    index_Q_batch = index_Q_batch_head // NUM_HEADS
    index_Q_head = index_Q_batch_head % NUM_HEADS

    offset_QKV_T = index_Q_batch.to(tl.int64) * stride_Q_B + index_Q_head.to(tl.int64) * stride_Q_H

    block_Q_ptr = tl.make_block_ptr(
        base=Q + offset_QKV_T,  # [None, None, 0, 0]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_T, stride_Q_D),
        offsets=(index_Q_block * BLOCK_SIZE_Q, 0),  # [None, None, index_Q_block*BLOCK_SIZE_Q, 0]
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),  # TODO: idk
    )

    # for K, V do not offset seq, The idea is to loop over all
    # K V blocks for single Q block
    block_V_ptr = tl.make_block_ptr(
        base=V + offset_QKV_T,  # [None, None, 0, 0]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_T, stride_V_D),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),  # TODO: idk
    )

    block_K_ptr = tl.make_block_ptr(
        base=K + offset_QKV_T,  # [None, None, 0, 0]
        shape=(HEAD_DIM, SEQ_LEN),
        # we will load K as K^T
        strides=(stride_K_D, stride_K_T),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),  # TODO: idk
    )

    # Output block will share same shape as Q
    block_O_ptr = tl.make_block_ptr(
        base=O + offset_QKV_T,  # [None, None, 0, 0]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_T, stride_O_D),
        offsets=(index_Q_block * BLOCK_SIZE_Q, 0),  # [None, None, index_Q_block*BLOCK_SIZE_Q, 0]
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),  # TODO: idk
    )

    # offset for Q over seq length: [index_Q_block, index_Q_block+1, ...., index_Q_block+BLOCK_SIZE_Q]
    offset_Q_T = index_Q_block * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offset for K and V over seq length: [0, 1, 2, ...., BLOCK_SIZE_KV-1]
    offset_KV_T = tl.arange(0, BLOCK_SIZE_KV)

    # running max for each row (or each query)
    # init to -inf
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # running norm factor L for each row again
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # accumulator for final output block in SMEM
    # idea is to keep an accumulator at SMEM and the dump it to block_O_ptr in GMEM
    block_O = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    block_Q = tl.load(block_Q_ptr)

    if STAGE == 1 or STAGE == 3:
        # first forward pass that handles lower triangle attention calculation
        # and it is common for both causal and non-causal
        block_O, l_i, m_i = _attn_forward_internal(
            block_O=block_O,
            l_i=l_i,
            m_i=m_i,
            block_Q=block_Q,
            block_K_ptr=block_K_ptr,
            block_V_ptr=block_V_ptr,
            index_Q_block=index_Q_block,
            tau=tau,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_KV=BLOCK_SIZE_KV,
            STAGE=4 - STAGE,
            offset_Q_T=offset_Q_T,
            offset_KV_T=offset_KV_T,
            SEQ_LEN=SEQ_LEN,
        )

    if STAGE == 3:
        block_O, l_i, m_i = _attn_forward_internal(
            block_O=block_O,
            l_i=l_i,
            m_i=m_i,
            block_Q=block_Q,
            block_K_ptr=block_K_ptr,
            block_V_ptr=block_V_ptr,
            index_Q_block=index_Q_block,
            tau=tau,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_KV=BLOCK_SIZE_KV,
            STAGE=2,
            offset_Q_T=offset_Q_T,
            offset_KV_T=offset_KV_T,
            SEQ_LEN=SEQ_LEN,
        )

    # logSumExp: m_i + log(l_i)
    m_i += tl.math.log(l_i)
    block_M_ptrs = M + index_Q_batch_head * SEQ_LEN + offset_Q_T
    tl.store(block_M_ptrs, m_i)

    block_O = block_O / l_i[:, None]
    # internal computations might have elevated type of block_O to say float32,
    # save it as original type
    tl.store(block_O_ptr, block_O.to(O.type.element_ty))


# torch treats autograd functions just like derivable funcs,
# passes dO and expects dQ, dK, dV
class FlashAttention(torch.autograd.Function):
    # conventions:
    # D -> head dimension
    # H -> number of heads

    @staticmethod
    def forward(ctx, Q, K, V, causal, tau) -> torch.Tensor:
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        # output O will be of same shape as Q [B, H, T, D]
        O = torch.empty_like(Q)

        stage = 3 if causal else 1

        # Fixed block sizes (removed autotune overhead)
        BLOCK_SIZE_Q = 64
        BLOCK_SIZE_KV = 32

        grid = (triton.cdiv(SEQ_LEN, BLOCK_SIZE_Q), BATCH_SIZE * NUM_HEADS)

        # M is logSumExp: TODO: exact formula for logSumExp
        M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32)

        _attn_forward_kernel[grid](  # type: ignore
            Q=Q,
            K=K,
            V=V,
            tau=tau,
            M=M,
            O=O,
            stride_Q_B=Q.stride(0),
            stride_Q_H=Q.stride(1),
            stride_Q_T=Q.stride(2),
            stride_Q_D=Q.stride(3),
            stride_K_B=K.stride(0),
            stride_K_H=K.stride(1),
            stride_K_T=K.stride(2),
            stride_K_D=K.stride(3),
            stride_V_B=V.stride(0),
            stride_V_H=V.stride(1),
            stride_V_T=V.stride(2),
            stride_V_D=V.stride(3),
            stride_O_B=O.stride(0),
            stride_O_H=O.stride(1),
            stride_O_T=O.stride(2),
            stride_O_D=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_KV=BLOCK_SIZE_KV,
            STAGE=stage,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.tau = tau
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal

        return O

    @staticmethod
    def backward(ctx, dO):  # type: ignore

        Q, K, V, O, M = ctx.saved_tensors

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        NUM_WARPS, NUM_STAGES = 4, 2
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 64

        # calculating D (per row) which will be used to calculate dS & then dQ, dV
        # D = rowsum(dO · O) note: its hadamard product
        p_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)  # [BATCH_SIZE, NUM_HEADS, SEQ_LEN]

        _attn_backward_preprocess_D[p_grid](  # type: ignore
            O=O, dO=dO, D=D, SEQ_LEN=SEQ_LEN, BLOCK_SIZE_Q=BLOCK_SIZE_MACRO, HEAD_DIM=HEAD_DIM
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)
        stage = 3 if ctx.causal else 1

        _attn_backward_dK_dV[grid](  # type: ignore
            Q=Q,
            K=K,
            V=V,
            tau=ctx.tau,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MICRO,
            BLOCK_SIZE_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        _attn_backward_dQ[grid](  # type: ignore
            Q=Q,
            K=K,
            V=V,
            tau=ctx.tau,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            BLOCK_SIZE_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        return dQ, dK, dV, None, None


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None  # type: ignore
    ref_dK, K.grad = K.grad.clone(), None  # type: ignore
    ref_dQ, Q.grad = Q.grad.clone(), None  # type: ignore

    # triton implementation
    tri_out = FlashAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None  # type: ignore
    tri_dK, K.grad = K.grad.clone(), None  # type: ignore
    tri_dQ, Q.grad = Q.grad.clone(), None  # type: ignore

    # compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)

    # Benchmark
    ms = triton.testing.do_bench(
        lambda: FlashAttention.apply(Q, K, V, causal, softmax_scale).half().backward(dO)
    )

    # FLOPs for attention: 2 * B * H * T^2 * D (QK^T) + 2 * B * H * T^2 * D (softmax*V) = 4*B*H*T^2*D per pass
    # Forward + Backward ≈ 2.5x forward FLOPs (backward computes dQ, dK, dV)
    flops = 2.5 * 4 * BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"Time: {ms:.3f}ms, TFLOPS: {tflops:.2f}")


if __name__ == "__main__":
    # test_op(BATCH_SIZE=1, NUM_HEADS=2, SEQ_LEN=64, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
    print("PASSED")
