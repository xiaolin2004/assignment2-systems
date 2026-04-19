import subprocess
import os
import sys

# Define model configurations from Table 1
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

CONTEXT_LENGTHS = [128, 256, 512, 1024]
VOCAB_SIZE = 10000
BATCH_SIZE = 4
WARMUP_STEPS = 5
NUM_STEPS = 10

def run_profiling():
    # Create profiles directory if it doesn't exist
    os.makedirs("profiles", exist_ok=True)
    
    python_executable = sys.executable

    for size_name, config in MODEL_CONFIGS.items():
        for context_len in CONTEXT_LENGTHS:
            print(f"Profiling {size_name} model with context length {context_len}...")
            
            output_filename = f"profiles/{size_name}_{context_len}"
            
            # Construct the command
            # nsys profile --trace=nvtx,cuda --output=... --force-overwrite=true python scripts/benchmark.py ...
            cmd = [
                "nsys", "profile",
                "--trace=nvtx,cuda",
                f"--output={output_filename}",
                "--force-overwrite=true",
                python_executable, "scripts/benchmark.py",
                "--vocab-size", str(VOCAB_SIZE),
                "--context-length", str(context_len),
                "--d-model", str(config["d_model"]),
                "--num-layers", str(config["num_layers"]),
                "--num-heads", str(config["num_heads"]),
                "--d-ff", str(config["d_ff"]),
                "--batch-size", str(BATCH_SIZE),
                "--warmup-steps", str(WARMUP_STEPS),
                "--num-steps", str(NUM_STEPS),
                "--backward",
                "--optimizer"
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error profiling {size_name} with context {context_len}: {e}")
                print("Continuing to next configuration...")
            except FileNotFoundError:
                print("Error: 'nsys' command not found. Make sure you are running on a machine with Nsight Systems installed.")
                # For testing purposes without nsys, we might want to just run the python script
                # uncomment the following lines to fallback to just running the script
                # print("Falling back to running python script directly...")
                # subprocess.run(cmd[6:], check=True)
                return

if __name__ == "__main__":
    run_profiling()
