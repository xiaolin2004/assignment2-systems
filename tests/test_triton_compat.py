import pytest

from cs336_systems.triton import describe_triton_unavailability, require_triton, triton_is_available
from cs336_systems.triton.weighted_sum import weighted_sum_fwd


def test_triton_kernel_modules_import_locally():
    assert weighted_sum_fwd is not None


def test_triton_unavailability_message_is_actionable():
    if triton_is_available():
        pytest.skip("This check only applies when Triton is unavailable locally.")

    with pytest.raises(RuntimeError, match="Use this machine for authoring and CPU/MPS validation"):
        require_triton()

    assert "Detected platform:" in describe_triton_unavailability()
