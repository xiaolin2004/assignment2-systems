import torch
import timeit
import sys
import math
import contextlib

# Handle local env without CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    synchronize = torch.cuda.synchronize
    memory_allocated = torch.cuda.memory_allocated
    reset_peak_memory_stats = torch.cuda.reset_peak_memory_stats
    empty_cache = torch.cuda.empty_cache
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    synchronize = torch.mps.synchronize
    memory_allocated = torch.mps.current_allocated_memory
    # MPS doesn't exactly match CUDA memory stats API, we'll do our best or mock
    # reset_peak_memory_stats not available on MPS usually
    reset_peak_memory_stats = lambda: None
    empty_cache = torch.mps.empty_cache
else:
    device = torch.device("cpu")
    synchronize = lambda: None
    memory_allocated = lambda: 0
    reset_peak_memory_stats = lambda: None
    empty_cache = lambda: None

from cs336_systems.model import scaled_dot_product_attention

def benchmark_attention():
    BATCH_SIZE = 8
    
    # Define compiled version
    # Note: torch.compile works best on CUDA. On MPS/CPU it might fallback or error.
    # We'll try it.
    try:
        compiled_attention = torch.compile(scaled_dot_product_attention)
        has_compile = True
    except Exception as e:
        print(f"Warning: torch.compile failed: {e}")
        has_compile = False

    configs = []
    # Cartesian product of [16, 32, 64, 128] and [256, 1024, 4096, 8192, 16384]
    # Reduced set for quick testing if needed, but per prompt we stick to full:
    for d in [16, 32, 64, 128]:
        for s in [256, 1024, 4096, 8192, 16384]:
            configs.append((d, s))
            
    print(f"{'d_k':<5} {'Seq':<6} | {'Eager Fwd':<10} {'Eager Bwd':<10} | {'Comp Fwd':<10} {'Comp Bwd':<10} | {'Fwd Spd':<8} {'Bwd Spd':<8}")
    print("-" * 100)

    for d_k, seq_len in configs:
        row_data = {"d_k": d_k, "seq": seq_len}
        
        # --- Eager Benchmark ---
        try:
            empty_cache()
            synchronize()
            Q = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, dtype=torch.float32, requires_grad=True)
            K = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, dtype=torch.float32, requires_grad=True)
            V = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, dtype=torch.float32, requires_grad=True)
            
            # Warmup Eager
            for _ in range(5):
                out = scaled_dot_product_attention(Q, K, V)
                out.sum().backward()
            synchronize()
            
            # Time Eager Fwd
            times = []
            for _ in range(20):
                synchronize()
                t0 = timeit.default_timer()
                out = scaled_dot_product_attention(Q, K, V)
                synchronize()
                times.append(timeit.default_timer() - t0)
            row_data["eager_fwd"] = sum(times)/len(times)*1000
            
            # Time Eager Bwd
            times = []
            grad = torch.randn_like(out)
            for _ in range(20):
                out = scaled_dot_product_attention(Q, K, V) # Re-compute for graph
                synchronize()
                t0 = timeit.default_timer()
                out.backward(grad)
                synchronize()
                times.append(timeit.default_timer() - t0)
                Q.grad=None; K.grad=None; V.grad=None
            row_data["eager_bwd"] = sum(times)/len(times)*1000
            
        except torch.cuda.OutOfMemoryError:
            row_data["eager_fwd"] = float('nan')
            row_data["eager_bwd"] = float('nan')
        except Exception as e:
            if "out of memory" in str(e).lower() or "buffer size" in str(e).lower():
                 row_data["eager_fwd"] = float('nan')
                 row_data["eager_bwd"] = float('nan')
            else:
                 print(f"Eager Error {d_k} {seq_len}: {e}")
                 continue

        # --- Compiled Benchmark ---
        if has_compile and not math.isnan(row_data.get("eager_fwd", 0)):
            try:
                empty_cache()
                synchronize()
                # New inputs not strictly needed but cleaner
                Q = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, dtype=torch.float32, requires_grad=True)
                K = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, dtype=torch.float32, requires_grad=True)
                V = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, dtype=torch.float32, requires_grad=True)
                
                # Warmup Compiled (Crucial for compilation)
                # Compilation happens here
                for _ in range(5):
                    out = compiled_attention(Q, K, V)
                    out.sum().backward()
                synchronize()
                
                # Time Comp Fwd
                times = []
                for _ in range(20):
                    synchronize()
                    t0 = timeit.default_timer()
                    out = compiled_attention(Q, K, V)
                    synchronize()
                    times.append(timeit.default_timer() - t0)
                row_data["comp_fwd"] = sum(times)/len(times)*1000
                
                # Time Comp Bwd
                times = []
                grad = torch.randn_like(out)
                for _ in range(20):
                    out = compiled_attention(Q, K, V)
                    synchronize()
                    t0 = timeit.default_timer()
                    # Backward of compiled module is also compiled
                    out.backward(grad)
                    synchronize()
                    times.append(timeit.default_timer() - t0)
                    Q.grad=None; K.grad=None; V.grad=None
                row_data["comp_bwd"] = sum(times)/len(times)*1000
                
            except Exception as e:
                # OOM or compile error
                row_data["comp_fwd"] = float('nan')
                row_data["comp_bwd"] = float('nan')
        else:
             row_data["comp_fwd"] = float('nan')
             row_data["comp_bwd"] = float('nan')

        # Formatting
        ef, eb = row_data.get("eager_fwd", float('nan')), row_data.get("eager_bwd", float('nan'))
        cf, cb = row_data.get("comp_fwd", float('nan')), row_data.get("comp_bwd", float('nan'))
        
        fwd_speedup = ef / cf if not math.isnan(ef) and not math.isnan(cf) and cf > 0 else 0.0
        bwd_speedup = eb / cb if not math.isnan(eb) and not math.isnan(cb) and cb > 0 else 0.0
        
        print(f"{d_k:<5} {seq_len:<6} | {ef:<10.4f} {eb:<10.4f} | {cf:<10.4f} {cb:<10.4f} | {fwd_speedup:<8.2f} {bwd_speedup:<8.2f}")

if __name__ == "__main__":
    benchmark_attention()
