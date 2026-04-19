import pytest
import torch

triton = pytest.importorskip("triton")

from cs336_systems.flash_attention.pytorch import FlashAttentionAutogradFunctionPyTorch
from cs336_systems.flash_attention.triton import flash_fwd_kernel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A CUDA GPU is required to run Triton forward kernels.",
)


def _extract_saved_lse(output: torch.Tensor, expected_shape: tuple[int, int]) -> torch.Tensor:
    saved_tensors = getattr(output.grad_fn, "saved_tensors", ())
    matches = [tensor for tensor in saved_tensors if tensor.shape == expected_shape]
    assert len(matches) == 1, f"Expected exactly one saved LSE tensor with shape {expected_shape}, found {len(matches)}."
    return matches[0]


def run_pytorch_standard(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = FlashAttentionAutogradFunctionPyTorch.apply(q_ref, k_ref, v_ref, is_causal)
    lse = _extract_saved_lse(out, (q.shape[0], q.shape[1]))
    return out.detach(), lse.detach()


def build_stepwise_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_tile_size: int,
    k_tile_size: int,
    is_causal: bool = False,
) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
    scale = q.shape[-1] ** -0.5
    q_tiles = q.split(q_tile_size, dim=1)
    k_tiles = k.split(k_tile_size, dim=1)
    v_tiles = v.split(k_tile_size, dim=1)

    snapshots: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    for q_block_idx, q_tile in enumerate(q_tiles):
        q_start = q_block_idx * q_tile_size
        running_o = torch.zeros_like(q_tile, dtype=torch.float32)
        running_l = torch.zeros(q_tile.shape[:2], device=q.device, dtype=torch.float32)
        running_m = torch.full_like(running_l, float("-inf"))

        for kv_block_idx, (k_tile, v_tile) in enumerate(zip(k_tiles, v_tiles)):
            k_start = kv_block_idx * k_tile_size
            scores = torch.matmul(q_tile.float(), k_tile.float().transpose(-1, -2)) * scale

            if is_causal:
                q_idx = torch.arange(q_tile.shape[1], device=q.device) + q_start
                k_idx = torch.arange(k_tile.shape[1], device=k.device) + k_start
                scores = scores.masked_fill(k_idx[None, None, :] > q_idx[None, :, None], float("-inf"))

            prev_m = running_m
            block_m = scores.max(dim=-1).values
            next_m = torch.maximum(prev_m, block_m)
            exp_scores = torch.exp(scores - next_m.unsqueeze(-1))
            rescale = torch.exp(prev_m - next_m)

            running_l = rescale * running_l + exp_scores.sum(dim=-1)
            running_o = rescale.unsqueeze(-1) * running_o + torch.matmul(exp_scores, v_tile.float())
            running_m = next_m

            normalized_o = (running_o / running_l.unsqueeze(-1)).to(q.dtype)
            lse = running_m + torch.log(running_l)
            snapshots[(q_block_idx, kv_block_idx)] = (normalized_o.detach().clone(), lse.detach().clone())

    return snapshots


def run_triton_flash_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_tile_size: int,
    k_tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.dtype == k.dtype == v.dtype == torch.float32
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]

    batch_size, n_queries, d_head = q.shape
    n_keys = k.shape[1]

    out = torch.empty_like(q)
    lse = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(n_queries, q_tile_size), batch_size)

    flash_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        n_queries,
        n_keys,
        d_head**-0.5,
        D=d_head,
        Q_TILE_SIZE=q_tile_size,
        K_TILE_SIZE=k_tile_size,
    )
    torch.cuda.synchronize()
    return out, lse


def make_inputs(
    batch_size: int,
    n_queries: int,
    n_keys: int,
    d_head: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    q = torch.randn(batch_size, n_queries, d_head, device="cuda", dtype=torch.float32, generator=generator)
    k = torch.randn(batch_size, n_keys, d_head, device="cuda", dtype=torch.float32, generator=generator)
    v = torch.randn(batch_size, n_keys, d_head, device="cuda", dtype=torch.float32, generator=generator)
    return q, k, v


@pytest.mark.parametrize(
    ("batch_size", "n_queries", "n_keys", "d_head", "q_tile_size", "k_tile_size"),
    [
        (2, 64, 64, 64, 16, 16),
        (2, 128, 64, 64, 32, 16),
    ],
)
def test_stepwise_reference_matches_pytorch_standard(
    batch_size: int,
    n_queries: int,
    n_keys: int,
    d_head: int,
    q_tile_size: int,
    k_tile_size: int,
):
    q, k, v = make_inputs(batch_size, n_queries, n_keys, d_head)

    ref_out, ref_lse = run_pytorch_standard(q, k, v, is_causal=False)
    stepwise = build_stepwise_reference(
        q,
        k,
        v,
        q_tile_size=q_tile_size,
        k_tile_size=k_tile_size,
        is_causal=False,
    )

    out_tiles = []
    lse_tiles = []
    num_q_blocks = n_queries // q_tile_size
    num_kv_blocks = n_keys // k_tile_size
    final_kv_block = num_kv_blocks - 1

    for q_block_idx in range(num_q_blocks):
        out_tile, lse_tile = stepwise[(q_block_idx, final_kv_block)]
        out_tiles.append(out_tile)
        lse_tiles.append(lse_tile)

    stepwise_out = torch.cat(out_tiles, dim=1)
    stepwise_lse = torch.cat(lse_tiles, dim=1)

    torch.testing.assert_close(stepwise_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(stepwise_lse, ref_lse, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("batch_size", "n_queries", "n_keys", "d_head", "q_tile_size", "k_tile_size"),
    [
        (2, 64, 64, 64, 16, 16),
        (2, 128, 64, 64, 32, 16),
    ],
)
def test_triton_matches_pytorch_stepwise(
    batch_size: int,
    n_queries: int,
    n_keys: int,
    d_head: int,
    q_tile_size: int,
    k_tile_size: int,
):
    q, k, v = make_inputs(batch_size, n_queries, n_keys, d_head)

    stepwise = build_stepwise_reference(
        q,
        k,
        v,
        q_tile_size=q_tile_size,
        k_tile_size=k_tile_size,
        is_causal=False,
    )

    num_q_blocks = n_queries // q_tile_size
    num_kv_blocks = n_keys // k_tile_size

    for q_block_idx in range(num_q_blocks):
        q_start = q_block_idx * q_tile_size
        q_tile = q[:, q_start : q_start + q_tile_size].contiguous()

        for kv_block_idx in range(num_kv_blocks):
            k_stop = (kv_block_idx + 1) * k_tile_size
            k_prefix = k[:, :k_stop].contiguous()
            v_prefix = v[:, :k_stop].contiguous()

            out_triton, lse_triton = run_triton_flash_forward(
                q_tile,
                k_prefix,
                v_prefix,
                q_tile_size=q_tile_size,
                k_tile_size=k_tile_size,
            )
            out_ref, lse_ref = stepwise[(q_block_idx, kv_block_idx)]

            try:
                torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(lse_triton, lse_ref, rtol=1e-3, atol=1e-3)
            except AssertionError as exc:
                raise AssertionError(
                    f"Mismatch at q_block={q_block_idx}, kv_blocks_seen={kv_block_idx + 1}. "
                    "This is the first prefix where Triton diverges from the PyTorch reference."
                ) from exc
