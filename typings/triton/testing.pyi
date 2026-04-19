from collections.abc import Callable, Sequence
from typing import Any, Protocol


class Benchmark:
    x_names: Sequence[str]
    x_vals: Sequence[Any]
    line_arg: str | None
    line_vals: Sequence[Any] | None
    line_names: Sequence[str] | None
    styles: Sequence[tuple[str, str]] | None
    ylabel: str | None
    plot_name: str | None
    args: dict[str, Any]

    def __init__(
        self,
        *,
        x_names: Sequence[str],
        x_vals: Sequence[Any],
        line_arg: str | None = ...,
        line_vals: Sequence[Any] | None = ...,
        line_names: Sequence[str] | None = ...,
        styles: Sequence[tuple[str, str]] | None = ...,
        ylabel: str | None = ...,
        plot_name: str | None = ...,
        args: dict[str, Any] | None = ...,
    ) -> None: ...


class PerfReport(Protocol):
    def run(
        self,
        *,
        show_plots: bool = ...,
        print_data: bool = ...,
        save_path: str = ...,
    ) -> Any: ...


def perf_report(benchmarks: Benchmark | Sequence[Benchmark]) -> Callable[[Callable[..., Any]], PerfReport]: ...
def do_bench(
    fn: Callable[[], Any],
    *,
    warmup: int = ...,
    rep: int = ...,
    grad_to_none: Sequence[Any] | None = ...,
    quantiles: Sequence[float] | None = ...,
    return_mode: str = ...,
) -> float | tuple[float, float, float]: ...
def assert_close(
    x: Any,
    y: Any,
    *,
    atol: float = ...,
    rtol: float = ...,
    err_msg: str = ...,
) -> None: ...
