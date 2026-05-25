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
        
        # STAGE 2 means we are at transition diagonal, be careful to apply mask 
        if STAGE = 2:
            mask = offset_Q_T[:, None] >= (start_kv + offset_KV_T[None, :]) 



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
    block_Q = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    block_Q = tl.load(block_Q_ptr)

    if STAGE == 1 or STAGE == 3:
        # first forward pass that handles lower triangle attention calculation
        # and it is common for both causal and non-causal
        
        


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
        

        

    @staticmethod
    def backward(ctx, dO) :
        ...
    





