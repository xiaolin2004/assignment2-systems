import argparse
from dataclasses import dataclass
from typing import Callable

import torch
import triton.testing

from cs336_systems.flash_attention.triton import FlashAttentionAutogradFunctionTriton


AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]


@dataclass
class BenchmarkResult:
    forward_ms: float | None
    backward_ms: float | None


def regular_pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        causal_mask = (
            torch.arange(q_len, device=q.device)[:, None]
            >= torch.arange(k_len, device=q.device)[None, :]
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    return FlashAttentionAutogradFunctionTriton.apply(q, k, v, is_causal)


def make_inputs(
    batch_size: int,
    seq_len: int,
    d_head: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, seq_len, d_head, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(batch_size, seq_len, d_head, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(batch_size, seq_len, d_head, device=device, dtype=dtype, requires_grad=requires_grad)
    do = torch.randn(batch_size, seq_len, d_head, device=device, dtype=dtype)
    return q, k, v, do


def benchmark_forward(
    fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    warmup: int,
    rep: int,
) -> float:
    return triton.testing.do_bench(
        lambda: fn(q, k, v, is_causal),
        warmup=warmup,
        rep=rep,
    )


def supports_backward(
    fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    is_causal: bool,
) -> tuple[bool, str | None]:
    try:
        out = fn(q, k, v, is_causal)
        out.backward(do)
        return True, None
    except Exception as exc:  # pragma: no cover - probe only
        return False, str(exc)


def benchmark_backward(
    fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    is_causal: bool,
    warmup: int,
    rep: int,
) -> float:
    out = fn(q, k, v, is_causal)
    return triton.testing.do_bench(
        lambda: out.backward(do, retain_graph=True),
        warmup=warmup,
        rep=rep,
        grad_to_none=[q, k, v],
    )


def benchmark_impl(
    name: str,
    fn: AttentionFn,
    batch_size: int,
    seq_len: int,
    d_head: int,
    dtype: torch.dtype,
    device: torch.device,
    is_causal: bool,
    warmup: int,
    rep: int,
) -> tuple[BenchmarkResult, str | None]:
    q_fwd, k_fwd, v_fwd, _ = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        d_head=d_head,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    try:
        forward_ms = benchmark_forward(fn, q_fwd, k_fwd, v_fwd, is_causal, warmup, rep)
    except Exception as exc:  # pragma: no cover - benchmark environment dependent
        return BenchmarkResult(forward_ms=None, backward_ms=None), f"{name} forward failed: {exc}"

    q_probe, k_probe, v_probe, do_probe = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        d_head=d_head,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    has_backward, backward_error = supports_backward(fn, q_probe, k_probe, v_probe, do_probe, is_causal)
    if not has_backward:
        return BenchmarkResult(forward_ms=forward_ms, backward_ms=None), (
            f"{name} backward unavailable: {backward_error}"
        )

    q_bwd, k_bwd, v_bwd, do_bwd = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        d_head=d_head,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    try:
        backward_ms = benchmark_backward(fn, q_bwd, k_bwd, v_bwd, do_bwd, is_causal, warmup, rep)
    except Exception as exc:  # pragma: no cover - benchmark environment dependent
        return BenchmarkResult(forward_ms=forward_ms, backward_ms=None), f"{name} backward failed: {exc}"
    return BenchmarkResult(forward_ms=forward_ms, backward_ms=backward_ms), None


def format_ms(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:8.3f}"


def format_speedup(baseline_ms: float | None, candidate_ms: float | None) -> str:
    if baseline_ms is None or candidate_ms is None or candidate_ms == 0:
        return "N/A"
    return f"{baseline_ms / candidate_ms:7.2f}x"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark regular PyTorch attention against the Triton FlashAttention-2 "
            "implementation using triton.testing.do_bench."
        )
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    return parser.parse_args()


def get_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Triton benchmarks.")

    torch.manual_seed(0)
    dtype = get_dtype(args.dtype)
    device = torch.device("cuda")

    print(
        f"# device={device} dtype={args.dtype} batch_size={args.batch_size} "
        f"causal={args.causal} warmup={args.warmup} rep={args.rep}"
    )
    print(
        f"{'seq':>6} {'d':>6} | "
        f"{'pt_fwd':>8} {'tr_fwd':>8} {'fwd_spd':>8} | "
        f"{'pt_bwd':>8} {'tr_bwd':>8} {'bwd_spd':>8}"
    )
    print("-" * 70)

    for seq_len in args.seq_lens:
        for d_head in args.head_dims:
            regular_result, regular_note = benchmark_impl(
                name="regular_pytorch",
                fn=regular_pytorch_attention,
                batch_size=args.batch_size,
                seq_len=seq_len,
                d_head=d_head,
                dtype=dtype,
                device=device,
                is_causal=args.causal,
                warmup=args.warmup,
                rep=args.rep,
            )
            triton_result, triton_note = benchmark_impl(
                name="triton_flashattention",
                fn=triton_flash_attention,
                batch_size=args.batch_size,
                seq_len=seq_len,
                d_head=d_head,
                dtype=dtype,
                device=device,
                is_causal=args.causal,
                warmup=args.warmup,
                rep=args.rep,
            )

            print(
                f"{seq_len:6d} {d_head:6d} | "
                f"{format_ms(regular_result.forward_ms)} "
                f"{format_ms(triton_result.forward_ms)} "
                f"{format_speedup(regular_result.forward_ms, triton_result.forward_ms):>8} | "
                f"{format_ms(regular_result.backward_ms)} "
                f"{format_ms(triton_result.backward_ms)} "
                f"{format_speedup(regular_result.backward_ms, triton_result.backward_ms):>8}"
            )

            for note in (regular_note, triton_note):
                if note is not None:
                    print(f"  note: seq={seq_len} d={d_head}: {note}")


if __name__ == "__main__":
    main()
