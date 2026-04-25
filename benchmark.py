
import time
import numpy as np
from cortical_sheet_tensor import TensorizedCorticalSheet
from cortical_column import ColumnConfig

def run_benchmarks():
    cfg = ColumnConfig(d_in=16, d_h=32)
    rng = np.random.default_rng(42)
    
    print("--- Cortical Sheet Benchmark ---")
    for nc in [10, 50, 100, 250, 500]:
        sheet = TensorizedCorticalSheet(n_cols=nc, cfg=cfg, k_active=nc//5)
        start = time.perf_counter()
        for _ in range(50):
            sheet.step(rng.normal(0, 1, 16), learn=True)
        elapsed = (time.perf_counter() - start) / 50 * 1000
        print(f"Columns: {nc:3d} | Time/Step: {elapsed:5.1f}ms")

if __name__ == "__main__":
    run_benchmarks()
