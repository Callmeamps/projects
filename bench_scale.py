"""
Phase 10.3: Scaling Benchmark

Compare numpy vs PyTorch (CPU vs GPU) for 100-1000+ columns.
"""

import time
import numpy as np

def bench_numpy(n_cols, steps=50):
    from cortical_sheet_tensor import TensorizedCorticalSheet, ColumnConfig
    cfg = ColumnConfig(d_in=16, d_h=32)
    sheet = TensorizedCorticalSheet(n_cols=n_cols, cfg=cfg)

    x = np.random.normal(0, 1, 16)
    start = time.time()
    for _ in range(steps):
        sheet.step(x, learn=False)
    elapsed = time.time() - start
    return elapsed, steps / elapsed


def bench_pytorch(n_cols, steps=50, device="cuda"):
    try:
        from cortical_sheet_pytorch import PyTorchedCorticalSheet
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            print(f"  CUDA not available, skipping GPU test")
            return None, None

        cfg = ColumnConfig(d_in=16, d_h=32)
        sheet = PyTorchedCorticalSheet(n_cols=n_cols, cfg=cfg, device=device)

        x = torch.randn(16, device=sheet.device)
        start = time.time()
        for _ in range(steps):
            sheet.step(x, learn=False)
        elapsed = time.time() - start
        return elapsed, steps / elapsed
    except Exception as e:
        print(f"  PyTorch error: {e}")
        return None, None


if __name__ == "__main__":
    print("=== Scaling Benchmark: Numpy vs PyTorch ===\n")

    sizes = [100, 500, 1000]

    print(f"{'Cols':>6} | {'Numpy (s)':>10} | {'Numpy (sps)':>12} | {'GPU (s)':>10} | {'GPU (sps)':>12}")
    print("-" * 65)

    for n in sizes:
        print(f"{n:6d} | ", end="", flush=True)

        # Numpy
        elapsed_np, sps_np = bench_numpy(n)
        print(f"{elapsed_np:10.3f} | {sps_np:12.1f} | ", end="", flush=True)

        # PyTorch GPU
        elapsed_gpu, sps_gpu = bench_pytorch(n, device="cuda")
        if elapsed_gpu is not None:
            print(f"{elapsed_gpu:10.3f} | {sps_gpu:12.1f}")
        else:
            print(f"{'N/A':>10} | {'N/A':>12}")

    print("\n=== Benchmark Complete ===")
