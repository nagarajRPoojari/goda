from ast import Tuple
from operator import index
from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
from torch._C import dtype
import triton #  pyright: ignore[import] ofcourse triton is not there in mac 
import torch 
import triton.language as tl # pyright: ignore[import] ofcourse triton is not there in mac 


@triton.jit
def _attn_forawrd_internal(
    block_O,
    l_i,
    m_i,
    block_Q,
    block_K_ptr,
    block_V_ptr,
    index_Q_block,
    tau,
    BLOCK_SIZE_Q    : tl.constexpr,
    BLOCK_SIZE_KV   : tl.constexpr,
    STAGE           : tl.constexpr,
    offset_Q_T      : tl.constexpr,
    offset_KV_T     : tl.constexpr,
    SEQ_LEN         : tl.constexpr,
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
        lo = tl.multiple_of(BLOCK_SIZE_Q) # seems to be hack that allows triton to optimize better : TODO: figure out how ?
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

        block_QK_T = tl.dot(block_Q, block_K) # block K already transposed

        
        # STAGE 2 means we are at transition diagonal, be careful to apply mask 
        if STAGE == 2:
            # offset_Q_T                     : [12, 13, 14, ... 12 + BLOCK_SIZE_Q]
            # offset_Q_T[:, None]            : [[12], [13], [14], .. [12 + BLOCK_SIZE_Q]]
            # offset_KV_T                    : [10, 11, 12, ... 10 + BLOCK_SIZE_KV]
            # offset_KV_T[None, :]           : [[10, 11, 12, .. 10 + BLOCK_SIZE_KV]]
            mask = offset_Q_T[:, None] >= (start_kv + offset_KV_T[None, :]) 
            block_QK_T = block_QK_T * tau + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(block_QK_T, 1)) # row max
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
        O_block = tl.dot(block_P, block_V, block_O)

        m_i = m_ij

        block_K_ptr = tl.advance(block_K_ptr, (0, BLOCK_SIZE_KV))
        block_V_ptr = tl.advance(block_V_ptr, (BLOCK_SIZE_KV, 0))

    return block_O, l_i, m_i


# stride of ith dimension: how much step i need to take to move from j to j+1 in ith dim
#   it is simply product of its inner dims
@triton.jit
def _attn_forward_kernel(
    Q, # [B, H, T, D]    
    K, # [B, H, T, D]    
    V, # [B, H, T, D]    
    tau, # softmax scale , usuall 1/sqrt(D)
    M, # [B, H, T]
    O, # [B, H, T, D]
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
    BACTH_SIZE,
    NUM_HEAD: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):

    index_Q_block = tl.program_id(0)
    index_Q_batch_head = tl.program_id(1)
    index_Q_batch = index_Q_batch_head //  NUM_HEAD
    index_Q_head = index_Q_batch_head % NUM_HEAD
    
    qkv_offset = (index_Q_batch.to(tl.int64) * stride_Q_B + index_Q_head * stride_Q_H)

    block_Q_ptr = tl.make_block_ptr(
        base        =  Q + qkv_offset, # [None, None, 0, 0]
        shape       = (SEQ_LEN, HEAD_DIM),
        strides     = (stride_Q_T, stride_Q_D),
        offsets     = (index_Q_block * BLOCK_SIZE_Q, 0), # [None, None, index_Q_block*BLOCK_SIZE_Q, 0]
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM), 
        order       = (1, 0), # TODO: idk
    )

    
    # for K, V do not offset seq, The idea is to loop over all
    # K V blocks for single Q block
    block_V_ptr= tl.make_block_ptr(
        base        =  V + qkv_offset, # [None, None, 0, 0]
        shape       = (SEQ_LEN, HEAD_DIM),
        strides     = (stride_V_T, stride_V_D),
        offsets     = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order       = (1, 0), # TODO: idk
    )

    block_K_ptr = tl.make_block_ptr(
        base        =  K + qkv_offset, # [None, None, 0, 0]
        shape       = (HEAD_DIM, SEQ_LEN),
        # we will load K as K
        strides     = (stride_K_D, stride_K_T),
        offsets     = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order       = (1, 0), # TODO: idk
    )

    # Output block will share same shape as Q 
    block_O_ptr = tl.make_block_ptr(
        base        =  O + qkv_offset, # [None, None, 0, 0]
        shape       = (SEQ_LEN, HEAD_DIM),
        strides     = (stride_O_T, stride_O_D),
        offsets     = (index_Q_block * BLOCK_SIZE_Q, 0), # [None, None, index_Q_block*BLOCK_SIZE_Q, 0]
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order       = (1, 0), # TODO: idk
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
         block_O, l_i, m_i = _attn_forawrd_internal(
            block_O         = block_O,
            l_i             = l_i,
            m_i             = m_i,
            block_Q         =block_Q,
            block_K_ptr     = block_K_ptr,
            block_V_ptr     = block_V_ptr,
            index_Q_block   = index_Q_block,
            tau             = tau,
            BLOCK_SIZE_Q    = BLOCK_SIZE_Q,
            BLOCK_SIZE_KV   = BLOCK_SIZE_KV,
            STAGE           = 4 - STAGE,
            offset_Q_T      = offset_Q_T,
            offset_KV_T     = offset_KV_T,
            SEQ_LEN         =  SEQ_LEN,
        )

    if STAGE == 3:
        
        block_O, l_i, m_i = _attn_forawrd_internal(
            block_O         = block_O,
            l_i             = l_i,
            m_i             = m_i,
            block_Q         =block_Q,
            block_K_ptr     = block_K_ptr,
            block_V_ptr     = block_V_ptr,
            index_Q_block   = index_Q_block,
            tau             = tau,
            BLOCK_SIZE_Q    = BLOCK_SIZE_Q,
            BLOCK_SIZE_KV   = BLOCK_SIZE_KV,
            STAGE           = 4 - STAGE,
            offset_Q_T      = offset_Q_T,
            offset_KV_T     = offset_KV_T,
            SEQ_LEN         =  SEQ_LEN,
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

        grid = lambda args: (torch.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS)
        
        # M is logSumExp: TODO: exact formula for logSumExp
        M  = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32)
        

        _attn_forward_kernel[grid](
            Q=Q,
            K=K,
            V=V,
            tau=tau,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.tau = tau
        ctx.HEAD_DIM = ctx.HEAD_DIM
        ctx.causal = ctx.causal

        return O
        

    @staticmethod
    def backward(ctx, dO) :
        ...
    





