from pathlib import Path


def test_triton_stub_package_layout_exists():
    repo_root = Path(__file__).resolve().parent.parent

    assert (repo_root / "typings" / "triton" / "__init__.pyi").is_file()
    assert (repo_root / "typings" / "triton" / "language" / "__init__.pyi").is_file()
    assert (repo_root / "typings" / "triton" / "language" / "math.pyi").is_file()
    assert (repo_root / "typings" / "triton" / "runtime" / "__init__.pyi").is_file()
    assert (repo_root / "typings" / "triton" / "testing.pyi").is_file()


def test_triton_language_stub_covers_block_ptr_workflow():
    repo_root = Path(__file__).resolve().parent.parent
    stub_text = (repo_root / "typings" / "triton" / "language" / "__init__.pyi").read_text()

    assert "def make_block_ptr(" in stub_text
    assert "def advance(" in stub_text
    assert "def static_range(" in stub_text
    assert "def cdiv(" in stub_text
    assert "def atomic_add(" in stub_text
    assert "def atomic_cas(" in stub_text
    assert "from . import math" in stub_text


def test_triton_runtime_and_testing_stubs_cover_benchmarking_workflow():
    repo_root = Path(__file__).resolve().parent.parent
    top_level_stub = (repo_root / "typings" / "triton" / "__init__.pyi").read_text()
    runtime_stub = (repo_root / "typings" / "triton" / "runtime" / "__init__.pyi").read_text()
    testing_stub = (repo_root / "typings" / "triton" / "testing.pyi").read_text()

    assert "from . import language, runtime, testing" in top_level_stub
    assert "class JITFunction" in runtime_stub
    assert "class Autotuner" in runtime_stub
    assert "driver: _Driver" in runtime_stub
    assert "class Benchmark" in testing_stub
    assert "def perf_report(" in testing_stub
    assert "def do_bench(" in testing_stub
