import argparse
import timeit
import torch
import logging
import contextlib

# Handle NVTX for non-CUDA devices
if torch.cuda.is_available():
    import torch.cuda.nvtx as nvtx
else:
    class MockNvtx:
        @staticmethod
        @contextlib.contextmanager
        def range(msg):
            yield
    nvtx = MockNvtx

from cs336_systems.model import BasicsTransformerLM

def benchmark():
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM forward and backward passes.")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=128, help="Context length")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=1024, help="Feedforward dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Number of warm-up steps")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of measurement steps")
    parser.add_argument("--backward", action="store_true", help="Include backward pass in benchmarking")
    parser.add_argument("--optimizer", action="store_true", help="Include optimizer step in benchmarking")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision with BF16")
    parser.add_argument("--profile-memory", action="store_true", help="Enable CUDA memory profiling")
    parser.add_argument("--memory-snapshot-file", type=str, default="memory_snapshot.pickle", help="File to save memory snapshot")
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu' and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    if args.compile:
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    if args.optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Generate random batch of data
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).to(device)
    # Targets for loss calculation (if backward)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).to(device)

    # Context manager for mixed precision
    if args.mixed_precision:
        mp_context = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        print("Mixed precision (BF16) enabled.")
    else:
        # Actually, we want nullcontext for default behavior if we are just running normal standard precision
        # But wait, the original code didn't have a context manager for the whole step.
        # Let's import nullcontext
        from contextlib import nullcontext
        mp_context = nullcontext()

    # Warm-up
    print(f"Running {args.warmup_steps} warm-up steps...")
    for _ in range(args.warmup_steps):
        with nvtx.range("warmup_step"):
            with mp_context:
                logits = model(x)
                if args.backward or args.optimizer:
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            if args.backward or args.optimizer:
                # Backward should generally be outside autocast, but the loss computation should be inside.
                # PyTorch docs say: "Scalers are not necessary for bfloat16 mixed precision."
                # and backward is usually called on the loss tensor.
                loss.backward()
                if args.optimizer:
                    optimizer.step()
                model.zero_grad(set_to_none=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Start memory recording if enabled
    if args.profile_memory:
        print("Starting memory recording...")
        try:
             # max_entries=100000 is a reasonable default
            torch.cuda.memory._record_memory_history(max_entries=100000)
        except AttributeError:
             print("Warning: torch.cuda.memory._record_memory_history not found or not supported.")

    # Benchmark
    print(f"Running {args.num_steps} steps for benchmarking...")
    
    total_time = 0.0
    
    for _ in range(args.num_steps):
        start_time = timeit.default_timer()
        
        with nvtx.range("step"):
            with mp_context:
                with nvtx.range("forward"):
                    logits = model(x)
            
                if args.backward or args.optimizer:
                    with nvtx.range("loss"):
                        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            if args.backward or args.optimizer:
                with nvtx.range("backward"):
                    loss.backward()
                
                if args.optimizer:
                    with nvtx.range("optimizer"):
                        optimizer.step()
                
                with nvtx.range("zero_grad"):
                    model.zero_grad(set_to_none=True)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = timeit.default_timer()
        total_time += (end_time - start_time)
        
    avg_time = total_time / args.num_steps
    print(f"Average time per step: {avg_time:.6f} seconds")

    # Stop memory recording and save snapshot
    if args.profile_memory:
        try:
            print(f"Saving memory snapshot to {args.memory_snapshot_file}...")
            torch.cuda.memory._dump_snapshot(args.memory_snapshot_file)
            torch.cuda.memory._record_memory_history(enabled=None)
        except AttributeError:
            print("Warning: Failed to save memory snapshot (function not found).")
        except Exception as e:
            print(f"Error saving memory snapshot: {e}")

if __name__ == "__main__":
    benchmark()
