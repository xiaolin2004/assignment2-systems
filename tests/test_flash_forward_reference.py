import pytest
import torch

triton = pytest.importorskip("triton")

from cs336_systems.flash_attention.triton import flash_fwd_kernel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A CUDA GPU is required to run Triton forward kernels.",
)


def reference_flash_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale

    if is_causal:
        q_idx = torch.arange(q.shape[-2], device=q.device)[:, None]
        k_idx = torch.arange(k.shape[-2], device=k.device)[None, :]
        scores = scores.masked_fill(k_idx > q_idx, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.float()).to(q.dtype)
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


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
        (1, 32, 32, 64, 16, 16),
        (2, 64, 64, 64, 16, 16),
        (2, 64, 96, 64, 32, 32),
        (3, 128, 64, 32, 32, 16),
    ],
)
def test_flash_forward_matches_reference(
    batch_size: int,
    n_queries: int,
    n_keys: int,
    d_head: int,
    q_tile_size: int,
    k_tile_size: int,
):
    q, k, v = make_inputs(batch_size, n_queries, n_keys, d_head)

    out_ref, lse_ref = reference_flash_forward(q, k, v)
    out_triton, lse_triton = run_triton_flash_forward(
        q,
        k,
        v,
        q_tile_size=q_tile_size,
        k_tile_size=k_tile_size,
    )

    assert torch.isfinite(out_triton).all(), "Triton forward produced non-finite values in O."
    assert torch.isfinite(lse_triton).all(), "Triton forward produced non-finite values in LSE."
    torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse_triton, lse_ref, rtol=1e-3, atol=1e-3)
