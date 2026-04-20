import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

# float32 sizes: 1MB, 10MB, 100MB, 1GB
MB = 1024 * 1024
DATA_SIZES = {
    "1MB":   1 * MB // 4,
    "10MB":  10 * MB // 4,
    "100MB": 100 * MB // 4,
    "1GB":   1024 * MB // 4,
}

WORLD_SIZES = [2, 4, 6]
WARMUP = 5
ITERATIONS = 20


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def benchmark_worker(rank, world_size, size_name, num_elements, result_queue):
    setup(rank, world_size)
    tensor = torch.ones(num_elements, dtype=torch.float32)

    # warmup
    for _ in range(WARMUP):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # timed runs
    dist.barrier()
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    dist.barrier()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / ITERATIONS * 1000
    # bus bandwidth: 2*(n-1)/n * size / time (ring all-reduce formula)
    size_bytes = num_elements * 4
    bus_bw_gbps = 2 * (world_size - 1) / world_size * size_bytes / (elapsed / ITERATIONS) / 1e9

    if rank == 0:
        result_queue.put((world_size, size_name, avg_ms, bus_bw_gbps))

    cleanup()


def run_benchmark(world_size, size_name, num_elements):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=benchmark_worker,
            args=(rank, world_size, size_name, num_elements, q),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    return q.get() if not q.empty() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="allreduce_results.npz")
    parser.add_argument("--plot", default="allreduce_benchmark.png")
    args = parser.parse_args()

    results = []
    print(f"{'World':>6} {'Size':>7} {'Avg(ms)':>10} {'BusBW(GB/s)':>13}")
    print("-" * 42)

    for world_size in WORLD_SIZES:
        for size_name, num_elements in DATA_SIZES.items():
            result = run_benchmark(world_size, size_name, num_elements)
            if result:
                ws, sn, avg_ms, bw = result
                print(f"{ws:>6} {sn:>7} {avg_ms:>10.2f} {bw:>13.2f}")
                results.append(result)

    # save raw results
    if results:
        ws_arr = np.array([r[0] for r in results])
        sn_arr = np.array([r[1] for r in results])
        ms_arr = np.array([r[2] for r in results])
        bw_arr = np.array([r[3] for r in results])
        np.savez(args.output, world_size=ws_arr, size_name=sn_arr,
                 avg_ms=ms_arr, bus_bw_gbps=bw_arr)
        print(f"\nResults saved to {args.output}")
        plot_results(results, args.plot)


def plot_results(results, outfile):
    import matplotlib.pyplot as plt

    size_order = list(DATA_SIZES.keys())
    world_sizes_seen = sorted(set(r[0] for r in results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {2: "tab:blue", 4: "tab:orange", 6: "tab:green"}
    markers = {2: "o", 4: "s", 6: "^"}

    for ws in world_sizes_seen:
        ws_results = {r[1]: r for r in results if r[0] == ws}
        x_labels = [s for s in size_order if s in ws_results]
        x = range(len(x_labels))
        ms_vals = [ws_results[s][2] for s in x_labels]
        bw_vals = [ws_results[s][3] for s in x_labels]

        ax1.plot(x, ms_vals, marker=markers[ws], color=colors[ws],
                 label=f"{ws} GPUs", linewidth=2, markersize=8)
        ax2.plot(x, bw_vals, marker=markers[ws], color=colors[ws],
                 label=f"{ws} GPUs", linewidth=2, markersize=8)

    for ax, ylabel, title in [
        (ax1, "Latency (ms)", "All-Reduce Latency"),
        (ax2, "Bus Bandwidth (GB/s)", "All-Reduce Bus Bandwidth"),
    ]:
        ax.set_xticks(range(len(size_order)))
        ax.set_xticklabels(size_order)
        ax.set_xlabel("Data Size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log") if ylabel.startswith("Latency") else None

    fig.suptitle("NCCL All-Reduce Benchmark (single node)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {outfile}")


if __name__ == "__main__":
    main()
