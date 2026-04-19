import subprocess
import sys
import os

# Define configurations
MODEL_CONFIG = {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12} # "small" config as requested for test
# MODEL_CONFIG = {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32} # "2.7B" config (commented out as per instructions for local run)

CONTEXT_LENGTHS = [128, 256, 512]
BATCH_SIZE = 4 # Default batch size
WARMUP_STEPS = 5
NUM_STEPS = 10 # Keep it short for profiling

def run_profile(context_len, backward, optimizer, filename_prefix, mixed_precision):
    cmd = [
        sys.executable, "scripts/benchmark.py",
        "--vocab-size", "10000",
        "--context-length", str(context_len),
        "--d-model", str(MODEL_CONFIG["d_model"]),
        "--num-layers", str(MODEL_CONFIG["num_layers"]),
        "--num-heads", str(MODEL_CONFIG["num_heads"]),
        "--d-ff", str(MODEL_CONFIG["d_ff"]),
        "--batch-size", str(BATCH_SIZE),
        "--warmup-steps", str(WARMUP_STEPS),
        "--num-steps", str(NUM_STEPS),
        "--profile-memory",
        "--memory-snapshot-file", f"{filename_prefix}_{context_len}_{'mp' if mixed_precision else 'fp32'}.pickle"
    ]
    
    if mixed_precision:
        cmd.append("--mixed-precision")
    
    if backward:
        cmd.append("--backward")
    if optimizer:
        cmd.append("--optimizer")
        
    print(f"Running profile for context {context_len}, precision={'Mixed (BF16)' if mixed_precision else 'FP32'}, backward={backward}, optimizer={optimizer}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Snapshot saved to {filename_prefix}_{context_len}_{'mp' if mixed_precision else 'fp32'}.pickle")
    except subprocess.CalledProcessError as e:
        print(f"Error profiling context {context_len}: {e}")

def main():
    os.makedirs("memory_profiles", exist_ok=True)
    
    precisions = [False, True] # False = FP32, True = Mixed Precision
    
    # 1. Forward Only
    print("\n--- Profiling Forward Pass Only ---")
    for mixed_precision in precisions:
        for ctx in CONTEXT_LENGTHS:
            run_profile(ctx, backward=False, optimizer=False, filename_prefix="memory_profiles/forward_only", mixed_precision=mixed_precision)
        
    # 2. Full Training Step (Forward + Backward + Optimizer)
    print("\n--- Profiling Full Training Step ---")
    for mixed_precision in precisions:
        for ctx in CONTEXT_LENGTHS:
            run_profile(ctx, backward=True, optimizer=True, filename_prefix="memory_profiles/full_step", mixed_precision=mixed_precision)

if __name__ == "__main__":
    main()
