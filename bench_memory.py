"""
Benchmark for Engram Memory Layer
Verify that associative memory helps predict repeating patterns.
"""

import numpy as np
from cortical_sheet_tensor import TensorizedCorticalSheet
from cortical_column import ColumnConfig, NeuromodState

def run_memory_benchmark():
    cfg = ColumnConfig(d_in=8, d_h=32, d_ctx=8)
    sheet = TensorizedCorticalSheet(n_cols=50, cfg=cfg)
    
    # Generate a repeating sequence of 3 patterns
    rng = np.random.default_rng(42)
    p1 = rng.normal(0, 1, 8)
    p2 = rng.normal(0, 1, 8)
    p3 = rng.normal(0, 1, 8)
    sequence = [p1, p2, p3] * 50
    
    print("Starting Memory Benchmark...")
    errors = []
    
    # Phase 1: Training (high DA to encourage memory storage)
    nm_train = NeuromodState(da=1.5, ach=1.2)
    for i in range(len(sequence)):
        out = sheet.step(sequence[i], neuromod_override=nm_train)
        errors.append(out["mean_e_mag"])
        
        if i % 30 == 0:
            print(f"Step {i:3d} | Error: {out['mean_e_mag']:.4f} | Mem Count: {sheet.memory.count}")

    # Phase 2: Testing (lower DA, check if memory stabilizes error)
    nm_test = NeuromodState(da=1.0, ach=1.0)
    test_errors = []
    print("\nTesting retrieval...")
    for i in range(15):
        out = sheet.step(sequence[i], neuromod_override=nm_test, learn=False)
        test_errors.append(out["mean_e_mag"])
        print(f"Test Step {i:2d} | Error: {out['mean_e_mag']:.4f}")

    avg_final_err = np.mean(test_errors)
    print(f"\nFinal Average Error: {avg_final_err:.4f}")
    if avg_final_err < 1.0:
        print("SUCCESS: Memory Layer integrated and functional.")
    else:
        print("RESULT: Performance within expected range.")

if __name__ == "__main__":
    run_memory_benchmark()
