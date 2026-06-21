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
    block_K = tl.load(  # [BLOCK_SIZE_KV, HEAD_DIM]
        K + offset_KV_T[:, None] * stride_seq + offset_D[None, :] * stride_dim
    )
    block_V = tl.load(  # [BLOCK_SIZE_KV, HEAD_DIM]
        V + offset_KV_T[:, None] * stride_seq + offset_D[None, :] * stride_dim
    )

    # For causal: Q at position q can only attend to K at position k <= q.
    # So dK[k] only gets contributions from Q blocks where max(q) >= k,
    # i.e. curr_offset_Q_T + BLOCK_SIZE_Q - 1 >= start_kv
    # => curr_offset_Q_T >= start_kv - BLOCK_SIZE_Q + 1
    # => first Q block to process: max(0, start_kv - BLOCK_SIZE_Q + 1) rounded to block boundary
    # => simply: start at the Q block that contains start_kv (all earlier Q blocks are future-only)
    # Causal: loop Q blocks from start_kv onwards (earlier Q rows can't see this KV block).
    # Non-causal: loop all Q blocks.
    if STAGE == 3:
        # causal: first Q block whose *last* row >= start_kv
        # that is q_block_idx * BLOCK_SIZE_Q + BLOCK_SIZE_Q - 1 >= start_kv
        # => q_block_idx >= (start_kv - BLOCK_SIZE_Q + 1) / BLOCK_SIZE_Q
        # => q_block_start = start_kv rounded down to Q block boundary
        q_start = (start_kv // BLOCK_SIZE_Q) * BLOCK_SIZE_Q
    else:
        q_start = 0

    curr_offset_Q_T = q_start

    for q_i in range((SEQ_LEN - q_start) // BLOCK_SIZE_Q):
        # 1. Update offset BEFORE loading
        offset_Q_T = curr_offset_Q_T + tl.arange(0, BLOCK_SIZE_Q)

        # 2. Dynamic pointer generation
        block_Qt_ptr = Q + offset_Q_T[None, :] * stride_seq + offset_D[:, None] * stride_dim
        block_dO_ptr = dO + offset_Q_T[:, None] * stride_seq + offset_D[None, :] * stride_dim

        # 3. Load regular tensors
        block_Qt = tl.load(block_Qt_ptr)
        m_i = tl.load(M + offset_Q_T).to(tl.float32)
        block_dO = tl.load(block_dO_ptr)
        block_D = tl.load(D + offset_Q_T).to(tl.float32)

        # 4. Compute raw attention scores matrix transpose: (QK^T)^T = K(Q^T)
        block_QKt = tau * tl.dot(block_K, block_Qt, out_dtype=tl.float32)
        block_Pt = tl.exp(block_QKt - m_i[None, :]).to(tl.float32)

        if STAGE == 3:
            # Causal: P^T[k, q] is nonzero only when q >= k
            # offset_KV_T: [BLOCK_SIZE_KV]  (k indices, fixed)
            # offset_Q_T:  [BLOCK_SIZE_Q]   (q indices, moving)
            # block_Pt shape: [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            mask = offset_Q_T[None, :] >= offset_KV_T[:, None]
            block_Pt = tl.where(mask, block_Pt, 0.0)

        # 5. Accumulate dV
        block_Pt_fp16 = block_Pt.to(tl.float16)
        block_dV = (block_dV + tl.dot(block_Pt_fp16, block_dO, out_dtype=tl.float32)).to(tl.float32)

        # 6. dP^T = V * dO^T
        block_dPt = tl.dot(block_V, tl.trans(block_dO), out_dtype=tl.float32).to(tl.float32)

        # 7. dS^T = P^T * (dP^T - D)
        block_dSt = (block_Pt * (block_dPt - block_D[None, :])).to(tl.float32)
        block_dSt_fp16 = block_dSt.to(tl.float16)

        # 8. Accumulate dK — tau factor because score = tau * Q @ K^T, so dK = tau * dS^T @ Q
        block_dK = (
            block_dK + tau * tl.dot(block_dSt_fp16, tl.trans(block_Qt), out_dtype=tl.float32)
        ).to(tl.float32)

        # 9. Step loop offset forward
        curr_offset_Q_T += BLOCK_SIZE_Q

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

    offset_D = tl.arange(0, HEAD_DIM)

    start_q = index_KV_block * BLOCK_SIZE_Q
    offset_Q_T = start_q + tl.arange(0, BLOCK_SIZE_Q)

    block_Q = tl.load(Q + offset_Q_T[:, None] * stride_seq + offset_D[None, :] * stride_dim)
    block_dQ = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    block_dO = tl.load(dO + offset_Q_T[:, None] * stride_seq + offset_D[None, :] * stride_dim)

    block_M = tl.load(M + offset_Q_T)
    block_M = block_M[:, None]

    Di = tl.load(D + offset_Q_T)

    # For causal: Q[q] only attends to K[k] where k <= q.
    # So dQ[q] only accumulates from KV blocks where min(k) <= max(q).
    # => only loop KV blocks up to and including the one containing start_q.
    if STAGE == 3:
        kv_end = start_q + BLOCK_SIZE_Q  # KV positions > start_q+BLOCK_SIZE_Q-1 are masked out
    else:
        kv_end = SEQ_LEN

    curr_offset_KV_T = 0

    block_Kt_ptr = (
        K + tl.arange(0, BLOCK_SIZE_KV)[None, :] * stride_seq + offset_D[:, None] * stride_dim
    )
    block_Vt_ptr = (
        V + tl.arange(0, BLOCK_SIZE_KV)[None, :] * stride_seq + offset_D[:, None] * stride_dim
    )

    for kv_i in range(kv_end // BLOCK_SIZE_KV):
        offset_KV_T = curr_offset_KV_T + tl.arange(0, BLOCK_SIZE_KV)

        block_Kt_ptr = K + offset_KV_T[None, :] * stride_seq + offset_D[:, None] * stride_dim
        block_Vt_ptr = V + offset_KV_T[None, :] * stride_seq + offset_D[:, None] * stride_dim

        block_Kt = tl.load(block_Kt_ptr)
        block_Vt = tl.load(block_Vt_ptr)

        block_QK = tau * tl.dot(block_Q, block_Kt)
        block_P = tl.math.exp(block_QK - block_M)

        if STAGE == 3:
            # Causal: P[q, k] nonzero only when q >= k
            mask = offset_Q_T[:, None] >= offset_KV_T[None, :]
            block_P = tl.where(mask, block_P, 0.0)

        # dP and dS
        block_dP = tl.dot(block_dO, block_Vt).to(tl.float32)
        block_dS = block_P * (block_dP - Di[:, None])
        block_dS = block_dS.to(tl.float16)

        block_dQ += tau * tl.dot(block_dS, tl.trans(block_Kt))

        curr_offset_KV_T += BLOCK_SIZE_KV

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
    if STAGE == 1:
        lo, hi = 0, index_Q_block * BLOCK_SIZE_Q
    elif STAGE == 2:
        lo, hi = index_Q_block * BLOCK_SIZE_Q, (index_Q_block + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        lo, hi = 0, SEQ_LEN

    block_K_ptr = tl.advance(block_K_ptr, (0, lo))
    block_V_ptr = tl.advance(block_V_ptr, (lo, 0))

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        block_K = tl.load(block_K_ptr)
        block_V = tl.load(block_V_ptr)

        block_QK_T = tl.dot(block_Q, block_K, out_dtype=tl.float32)

        if STAGE == 2:
            mask = offset_Q_T[:, None] >= (start_kv + offset_KV_T[None, :])
            block_QK_T = block_QK_T * tau + tl.where(mask, 0.0, -1.0e6)
            row_max = tl.max(block_QK_T, 1).to(tl.float32)
            m_ij = tl.maximum(m_i, row_max).to(tl.float32)
            block_QK_T = block_QK_T - m_ij[:, None]
        else:
            row_max = (tl.max(block_QK_T, 1) * tau).to(tl.float32)
            m_ij = tl.maximum(m_i, row_max).to(tl.float32)
            block_QK_T = (block_QK_T * tau) - m_ij[:, None]

        block_P = tl.exp(block_QK_T).to(tl.float32)
        l_ij = tl.sum(block_P, 1).to(tl.float32)
        alpha = tl.exp(m_i - m_ij).to(tl.float32)
        block_P = block_P.to(tl.float16)
        l_i = (l_i * alpha + l_ij).to(tl.float32)
        block_O = (block_O * alpha[:, None] + tl.dot(block_P, block_V, out_dtype=tl.float32)).to(
            tl.float32
        )
        m_i = m_ij.to(tl.float32)

        block_K_ptr = tl.advance(block_K_ptr, (0, BLOCK_SIZE_KV))
        block_V_ptr = tl.advance(block_V_ptr, (BLOCK_SIZE_KV, 0))

    return block_O, l_i, m_i


@triton.jit
def _attn_forward_kernel(
    Q,  # [B, H, T, D]
    K,  # [B, H, T, D]
    V,  # [B, H, T, D]
    tau,
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
        base=Q + offset_QKV_T,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_T, stride_Q_D),
        offsets=(index_Q_block * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    block_V_ptr = tl.make_block_ptr(
        base=V + offset_QKV_T,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_T, stride_V_D),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    block_K_ptr = tl.make_block_ptr(
        base=K + offset_QKV_T,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_D, stride_K_T),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    block_O_ptr = tl.make_block_ptr(
        base=O + offset_QKV_T,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_T, stride_O_D),
        offsets=(index_Q_block * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    offset_Q_T = index_Q_block * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offset_KV_T = tl.arange(0, BLOCK_SIZE_KV)

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    block_O = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    block_Q = tl.load(block_Q_ptr)

    if STAGE == 1 or STAGE == 3:
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

    m_i += tl.math.log(l_i)
    block_M_ptrs = M + index_Q_batch_head * SEQ_LEN + offset_Q_T
    tl.store(block_M_ptrs, m_i)

    block_O = block_O / l_i[:, None]
    tl.store(block_O_ptr, block_O.to(O.type.element_ty))


class FlashAttentionMQAKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, tau) -> torch.Tensor:
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        O = torch.empty_like(Q)

        stage = 3 if causal else 1

        # Halve block sizes for large HEAD_DIM to stay within 64KB SMEM (T4)
        BLOCK_SIZE_Q = 32 if HEAD_DIM > 64 else 64
        BLOCK_SIZE_KV = 16 if HEAD_DIM > 64 else 32

        grid = (triton.cdiv(SEQ_LEN, BLOCK_SIZE_Q), BATCH_SIZE * NUM_HEADS)

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
            num_stages=1,
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

        NUM_WARPS, NUM_STAGES = 4, 1
        # Halve block sizes for large HEAD_DIM to stay within 64KB SMEM (T4)
        BLOCK_SIZE_MICRO = 16 if HEAD_DIM > 64 else 32
        BLOCK_SIZE_MACRO = 32 if HEAD_DIM > 64 else 64

        p_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)  # [BATCH_SIZE, NUM_HEADS, SEQ_LEN]

        _attn_backward_preprocess_D[p_grid](  # type: ignore
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=HEAD_DIM,
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

    # Reference correctness check — skip if the [B,H,T,T] attention matrix won't fit in VRAM.
    # float32 attn matrix bytes = B * H * T * T * 4
    attn_matrix_gb = BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN * 4 / 1e9
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    skip_ref = attn_matrix_gb > free_gb * 0.5  # leave headroom for grads

    if skip_ref:
        print(f"  [skip reference: attn matrix ~{attn_matrix_gb:.1f}GB, free ~{free_gb:.1f}GB]")
    else:
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

        tri_out = FlashAttentionMQAKernel.apply(Q, K, V, causal, softmax_scale).half()
        tri_out.backward(dO)
        tri_dV, V.grad = V.grad.clone(), None  # type: ignore
        tri_dK, K.grad = K.grad.clone(), None  # type: ignore
        tri_dQ, Q.grad = Q.grad.clone(), None  # type: ignore

        rtol = 0.0
        atol = 1e-2
        assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol), "O mismatch"
        assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol), "dK mismatch"
        assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol), "dV mismatch"
        assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol), "dQ mismatch"

    # Benchmark (always runs)
    ms = triton.testing.do_bench(
        lambda: FlashAttentionMQAKernel.apply(Q, K, V, causal, softmax_scale).half().backward(dO)
    )

    flops = 2.5 * 4 * BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"Time: {ms:.3f}ms, TFLOPS: {tflops:.2f}")


if __name__ == "__main__":
    test_op(BATCH_SIZE=2, NUM_HEADS=2, SEQ_LEN=128, HEAD_DIM=64, causal=True)
    print("PASSED")
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
    print("PASSED (large)")
