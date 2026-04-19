from collections.abc import Callable, Sequence
from typing import Any, Protocol


class KernelInterface(Protocol):
    fn: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getitem__(self, grid: Any) -> Callable[..., Any]: ...


class JITFunction:
    fn: Callable[..., Any]
    arg_names: list[str]
    constexprs: tuple[int, ...] | None

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getitem__(self, grid: Any) -> Callable[..., Any]: ...
    def warmup(self, *args: Any, **kwargs: Any) -> Any: ...
    def parse(self) -> Any: ...


class Autotuner:
    fn: Callable[..., Any]
    configs: Sequence[Any]
    key: Sequence[str]
    best_config: Any

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getitem__(self, grid: Any) -> Callable[..., Any]: ...
    def warmup(self, *args: Any, **kwargs: Any) -> Any: ...


class InterpreterError(RuntimeError): ...


class MockTensor:
    dtype: Any
    shape: tuple[int, ...]

    def __init__(self, dtype: Any, shape: Sequence[int]) -> None: ...


class TensorWrapper:
    base: Any
    dtype: Any

    def __init__(self, base: Any, dtype: Any) -> None: ...


def reinterpret(tensor: Any, dtype: Any) -> Any: ...


class _Driver:
    active: Any
    default: Any


driver: _Driver
