import subprocess
import re
import sys
import contextlib

# Define model configurations matching Table 1 (same as run_profiling.py)
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

CONTEXT_LENGTH = 1024  # Fixed context length for this comparison as per convention or choose one
BATCH_SIZE = 4 # Matching run_profiling
WARMUP_STEPS = 5
NUM_STEPS = 10

def run_benchmark(config_name, config, compiled, backward=True, optimizer=False):
    cmd = [
        sys.executable, "scripts/benchmark.py",
        "--vocab-size", "10000",
        "--context-length", str(CONTEXT_LENGTH),
        "--d-model", str(config["d_model"]),
        "--num-layers", str(config["num_layers"]),
        "--num-heads", str(config["num_heads"]),
        "--d-ff", str(config["d_ff"]),
        "--batch-size", str(BATCH_SIZE),
        "--warmup-steps", str(WARMUP_STEPS),
        "--num-steps", str(NUM_STEPS),
    ]
    
    if compiled:
        cmd.append("--compile")
    
    if backward:
        cmd.append("--backward")
    if optimizer:
        cmd.append("--optimizer")
        
    mode_str = "Fwd"
    if backward: mode_str += "+Bwd"
    if optimizer: mode_str += "+Opt"
    
    print(f"Running {config_name} [{'Compiled' if compiled else 'Eager'}] [{mode_str}]...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse output for timing
        match = re.search(r"Average time per step: (\d+\.\d+) seconds", result.stdout)
        if match:
            return float(match.group(1))
        else:
            print(f"Error parsing output for {config_name}: {result.stdout}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark for {config_name}: {e.stderr}")
        return None

def main():
    print(f"{'Model Size':<15} {'Mode':<15} {'Eager (s)':<15} {'Compiled (s)':<15} {'Speedup':<10}")
    print("-" * 75)
    
    # Run only small model for compilation test if local
    # But per request "compile your entire Transformer model... How does the performance... change?"
    # We will try 'small' first.
    
    configs_to_run = ["small"] # Limit to small for local testing
    
    modes = [
        {"backward": False, "optimizer": False, "name": "Fwd"},
        {"backward": True,  "optimizer": False, "name": "Fwd+Bwd"},
        {"backward": True,  "optimizer": True,  "name": "Fwd+Bwd+Opt"},
    ]
    
    for size_name in configs_to_run:
        config = MODEL_CONFIGS[size_name]
        for mode in modes:
            t_eager = run_benchmark(size_name, config, compiled=False, backward=mode["backward"], optimizer=mode["optimizer"])
            t_compiled = run_benchmark(size_name, config, compiled=True, backward=mode["backward"], optimizer=mode["optimizer"])
            
            e_str = f"{t_eager:.4f}" if t_eager else "N/A"
            c_str = f"{t_compiled:.4f}" if t_compiled else "N/A"
            
            speedup = "N/A"
            if t_eager and t_compiled and t_compiled > 0:
                speedup = f"{t_eager / t_compiled:.2f}x"
                
            print(f"{size_name:<15} {mode['name']:<15} {e_str:<15} {c_str:<15} {speedup:<10}")
