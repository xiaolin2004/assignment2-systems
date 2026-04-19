from . import language, runtime, testing
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar, overload

_F = TypeVar("_F", bound=Callable[..., Any])


class Kernel(Protocol):
    fn: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getitem__(self, grid: Any) -> Callable[..., Any]: ...


class Config:
    kwargs: dict[str, Any]
    num_warps: int
    num_stages: int
    num_ctas: int
    maxnreg: int | None

    def __init__(
        self,
        kwargs: dict[str, Any],
        *,
        num_warps: int = ...,
        num_stages: int = ...,
        num_ctas: int = ...,
        maxnreg: int | None = ...,
        pre_hook: Callable[..., Any] | None = ...,
    ) -> None: ...


class OutOfResources(Exception): ...


@overload
def jit(
    fn: _F,
    /,
    *,
    version: str | None = ...,
    repr: Callable[..., str] | None = ...,
    launch_metadata: Callable[..., Any] | None = ...,
) -> Kernel: ...
@overload
def jit(
    *,
    version: str | None = ...,
    repr: Callable[..., str] | None = ...,
    launch_metadata: Callable[..., Any] | None = ...,
) -> Callable[[_F], Kernel]: ...

def autotune(
    configs: Sequence[Config],
    key: Sequence[str],
    *,
    prune_configs_by: dict[str, Any] | None = ...,
    reset_to_zero: Sequence[str] | None = ...,
    restore_value: Sequence[str] | None = ...,
    pre_hook: Callable[..., Any] | None = ...,
    post_hook: Callable[..., Any] | None = ...,
    warmup: int = ...,
    rep: int = ...,
    use_cuda_graph: bool = ...,
    do_bench: Callable[..., Any] | None = ...,
) -> Callable[[_F], Kernel]: ...

def heuristics(values: dict[str, Callable[..., Any]]) -> Callable[[_F], Kernel]: ...
def cdiv(x: int, y: int) -> int: ...
def next_power_of_2(n: int) -> int: ...
