import torch
from jaxtyping import Float
from torch import Tensor
from einops import einsum

# A torch.autograd.Function subclass that implements FlashAttention2 using only standard PyTorch operations (no Triton!).
class FlashAttentionAutogradFunctionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        q: Float[Tensor, "batch queries d_head"],
        k: Float[Tensor, "batch keys d_head"],
        v: Float[Tensor, "batch keys d_value"],
        is_causal: bool = False,
    ) -> Float[Tensor, "batch queries d_value"]:
        """
        Forward pass for FlashAttention2 using PyTorch operations.

        Args:
            ctx: Context object to save information for backward computation.
            q: Query tensor of shape (batch, queries, d_head)
            k: Key tensor of shape (batch, keys, d_head)
            v: Value tensor of shape (batch, keys, d_value)
        Returns:
            Output tensor of shape (batch, queries, d_value)
        """
        q_block_size, kv_block_size = 16, 16
        softmax_scale = q.shape[-1] ** -0.5

        # 沿着序列维分块，而不是沿 batch 维分块
        q_tiles = q.split(q_block_size, dim=1)
        k_tiles = k.split(kv_block_size, dim=1)
        v_tiles = v.split(kv_block_size, dim=1)

        output_tiles = []
        lse_tiles = []

        for q_tile in q_tiles:
            # 对当前 q tile 的每一行，维护三个 running state：
            # running_m: 到目前为止见过的最大 score，用来保证 softmax 数值稳定
            # running_l: 在 running_m 这个基准下的 exp 和
            # running_o: 尚未除以 running_l 的加权输出
            running_o = torch.zeros_like(q_tile)
            running_l = torch.zeros(q_tile.shape[:-1], device=q_tile.device, dtype=q_tile.dtype)
            # m 初始化为 -inf，这样第一块会自然退化成该块自己的 row max
            running_m = torch.full_like(running_l, float("-inf"))

            for k_tile, v_tile in zip(k_tiles, v_tiles):
                # 旧的 m 要先保留下来，后面需要用 exp(m_old - m_new) 对旧状态做重标定
                prev_m = running_m

                # 计算当前 q tile 和当前 k tile 的 pre-softmax attention scores
                scores = einsum(q_tile, k_tile, "b q d, b k d -> b q k") * softmax_scale
                # 当前块每一行的最大值
                block_m = scores.max(dim=-1).values
                # m_new 是“看到当前块之后”的全局行最大值
                next_m = torch.maximum(prev_m, block_m)

                # 在新的 m 基准下计算当前块的 exp(scores - m_new)
                exp_scores = torch.exp(scores - next_m.unsqueeze(-1))
                # 旧块累计量都要从旧基准 m_old 重标定到新基准 m_new
                row_rescale = torch.exp(prev_m - next_m)

                # l_j = exp(m_{j-1} - m_j) * l_{j-1} + rowsum(exp_scores)
                running_l = row_rescale * running_l + exp_scores.sum(dim=-1)
                # O_j = diag(exp(m_{j-1} - m_j)) * O_{j-1} + exp_scores @ V_j
                running_o = row_rescale.unsqueeze(-1) * running_o + exp_scores @ v_tile
                running_m = next_m

            # 最后再除以 running_l，得到真正归一化后的输出
            normalized_o = running_o / running_l.unsqueeze(-1)
            # L = logsumexp(scores) = m + log(l)
            lse = running_m + torch.log(running_l)

            output_tiles.append(normalized_o)
            lse_tiles.append(lse)

        # 将所有 q tiles 在 query 维拼回完整输出
        output = torch.cat(output_tiles, dim=1)
        # 将所有 tile 的 L 也在 query 维拼回去，形状应为 [batch, queries]
        lse = torch.cat(lse_tiles, dim=1)
        ctx.save_for_backward(q, k, v, output, lse)
        return output
