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
