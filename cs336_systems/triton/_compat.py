from __future__ import annotations

import platform

TRITON_IMPORT_ERROR: Exception | None = None

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - depends on local platform.
    triton = None
    tl = None
    TRITON_IMPORT_ERROR = exc


def triton_is_available() -> bool:
    return TRITON_IMPORT_ERROR is None


def describe_triton_unavailability() -> str:
    system = platform.system()
    machine = platform.machine()
    return (
        "Triton is unavailable in this environment. "
        f"Detected platform: {system} {machine}. "
        "Use this machine for authoring and CPU/MPS validation, then run Triton kernels on a Linux x86_64 CUDA host."
    )


def require_triton() -> None:
    if TRITON_IMPORT_ERROR is None:
        return

    raise RuntimeError(describe_triton_unavailability()) from TRITON_IMPORT_ERROR
