
import numpy as np
from cortical_hierarchy import CortexHierarchy
from cortical_column import ColumnConfig

def run_hierarchy_task():
    cfg = ColumnConfig(d_in=8, d_h=16)
    hier = CortexHierarchy(cfg=cfg)
    hier.add_area("V1", n_cols=20)
    hier.add_area("V2", n_cols=10)
    hier.link_ff("V1", "V2")
    hier.link_fb("V2", "V1")
    
    # Patterns
    A = np.random.normal(0.5, 0.1, 8)
    B = np.random.normal(-0.5, 0.1, 8)
    
    v1_stability = []
    v2_stability = []
    
    print("Running Hierarchical Stability Task...")
    prev_v1_h = None
    prev_v2_h = None
    
    for i in range(50):
        # 5 steps of A, then 5 steps of B
        target = A if (i // 5) % 2 == 0 else B
        # Add local noise to V1 frames
        frame = target + np.random.normal(0, 0.05, 8)
        
        out = hier.step(frame)
        
        v1_h = np.mean(hier.areas["V1"].h, axis=0)
        v2_h = np.mean(hier.areas["V2"].h, axis=0)
        
        if prev_v1_h is not None:
            v1_stability.append(np.linalg.norm(v1_h - prev_v1_h))
            v2_stability.append(np.linalg.norm(v2_h - prev_v2_h))
            
        prev_v1_h = v1_h
        prev_v2_h = v2_h
        
    avg_v1 = np.mean(v1_stability)
    avg_v2 = np.mean(v2_stability)
    
    print(f"Average V1 Change (Frame-to-Frame): {avg_v1:.4f}")
    print(f"Average V2 Change (Frame-to-Frame): {avg_v2:.4f}")
    
    if avg_v2 < avg_v1:
        print("PASSED: V2 is more temporally stable than V1")
    else:
        print("FAILED: V2 not stable enough")

if __name__ == "__main__":
    run_hierarchy_task()
