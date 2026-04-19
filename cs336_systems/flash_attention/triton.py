from numpy import tri
import torch
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr,K_ptr,V_ptr,
    O_ptr, L_ptr,
    stride_qb,stride_qq,stride_qd,
    stride_kb,stride_kk,stride_kd,
    stride_vb,stride_vk,stride_vd,
    stride_ob,stride_oq,stride_od,
    stride_lb,stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D:tl.constexpr,
    Q_TILE_SIZE:tl.constexpr,
    K_TILE_SIZE:tl.constexpr,
):
    # 指针参数：
    # Q_ptr, K_ptr, V_ptr:
    #   输入 Q/K/V 的起始地址，对应形状分别是 [B, N_QUERIES, D] / [B, N_KEYS, D] / [B, N_KEYS, D]
    # O_ptr:
    #   输出 attention result O 的起始地址，形状为 [B, N_QUERIES, D]
    # L_ptr:
    #   输出每个 query row 的 logsumexp，形状为 [B, N_QUERIES]
    #
    # stride 参数：
    # stride_qb, stride_qq, stride_qd:
    #   Q 在 batch/query/d_head 三个维度上的 stride
    # stride_kb, stride_kk, stride_kd:
    #   K 在 batch/key/d_head 三个维度上的 stride
    # stride_vb, stride_vk, stride_vd:
    #   V 在 batch/key/d_head 三个维度上的 stride
    # stride_ob, stride_oq, stride_od:
    #   O 在 batch/query/d_head 三个维度上的 stride
    # stride_lb, stride_lq:
    #   L 在 batch/query 两个维度上的 stride
    #
    # 运行时形状参数：
    # N_QUERIES:
    #   完整 query 序列长度
    # N_KEYS:
    #   完整 key/value 序列长度
    # scale:
    #   attention score 的缩放因子，通常为 1 / sqrt(D)
    #
    # constexpr / meta-parameters：
    # D:
    #   head dimension，编译期常量
    # Q_TILE_SIZE:
    #   每个 program 负责的 query tile 高度
    # K_TILE_SIZE:
    #   内层循环每次处理的 key/value tile 宽度
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # program_id(0): 第几个 query tile
    # program_id(1): 第几个 batch

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES,D),
        strides = (stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # 每个 program 从对应的 query tile 开始处理
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0), 
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS,D),
        strides = (stride_kk, stride_kd),
        offsets=(0, 0), # K 从头开始处理
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS,D),
        strides = (stride_vk, stride_vd),
        offsets=(0, 0), # V 从头开始处理
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES,D),
        strides = (stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # O 从对应的 query tile 开始写入
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides = (stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,), # L 从对应的 query tile 开始写入
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )   
    
    Q_tile = tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero") # shape: [Q_TILE_SIZE, D]
    O_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) # accumulator for output, shape: [Q_TILE_SIZE, D]
    L_tile = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # accumulator for logsumexp, shape: [Q_TILE_SIZE,]
    max_score_tile = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32) # accumulator for max score, shape: [Q_TILE_SIZE,]
    
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # 每次处理一个 key/value tile
        K_tile = tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero") # shape: [K_TILE_SIZE, D]
        V_tile = tl.load(V_block_ptr,boundary_check=(0,1),padding_option="zero") # shape: [K_TILE_SIZE, D]

        # 计算pre-softmax的attention score，shape: [Q_TILE_SIZE, K_TILE_SIZE]
        score_tile = tl.dot(Q_tile, tl.trans(K_tile))*scale # score_tile = Q_tile @ K_tile.T
        
        m_old = max_score_tile
        m_new = tl.maximum(m_old, tl.max(score_tile, axis=1)) # 每个 query row 的 max score，shape: [Q_TILE_SIZE,]
        p = tl.exp(score_tile - m_new[:, None]) # 计算每个 score 的 exp，并且为了数值稳定性减去 max score，shape: [Q_TILE_SIZE, K_TILE_SIZE]
        l_new = tl.exp(m_old-m_new)*L_tile + tl.sum(p, axis=1) # 更新 logsumexp，shape: [Q_TILE_SIZE,]
        O_tile = tl.exp(m_old - m_new)[:, None]*O_tile + tl.dot(p, V_tile) # 更新 output，shape: [Q_TILE_SIZE, D]
        max_score_tile = m_new
        L_tile = l_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0)) # 移动到下一个 key tile
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    O_tile = O_tile / L_tile[:, None] # 最后除以 logsumexp 得到最终的 attention output，shape: [Q_TILE_SIZE, D]
    tl.store(O_block_ptr, O_tile)
    L_tile = tl.log(L_tile) + max_score_tile # 最后计算 logsumexp 的值，shape: [Q_TILE_SIZE,]
    tl.store(L_block_ptr, L_tile)
    
class FlashAttentionAutogradFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v,is_causal=False):
        """
        q: [batch, queries, d_head]
        k: [batch, keys, d_head]
        v: [batch, keys, d_head]

        Returns:
        output: [batch, queries, d_head]
        lse: [batch, queries] 每个 query 行的 logsumexp，用于 backward pass 的数值稳定性
        """
        batch_size, n_queries, d_head = q.shape
        n_keys = k.shape[1]
        output = torch.empty_like(q)
        lse = torch.empty((batch_size, n_queries), device=q.device, dtype=q.dtype)
        
        # 这里 launch Triton kernel，block 和 grid 的配置需要根据输入形状和 tile size 来确定
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        grid = (triton.cdiv(n_queries, Q_TILE_SIZE), batch_size) # 每个 block 处理一个 query tile 和一个 batch
        flash_fwd_kernel[grid](
            q,k,v,
            output,lse,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            lse.stride(0), lse.stride(1),
            n_queries, n_keys,
            scale=1.0/d_head**0.5,
            D=d_head,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )
        
        ctx.save_for_backward(q,k,v,output,lse)
        return output