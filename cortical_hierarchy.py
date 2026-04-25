"""
Cortical Microcircuit — Phase 9.1: Multi-Area Hierarchy

Manages communication between multiple cortical sheets.
V1 (Sensory) <-> V2 (Abstract)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from cortical_sheet_tensor import TensorizedCorticalSheet
from cortical_column import ColumnConfig, NeuromodState

class CortexHierarchy:
    def __init__(self, cfg: Optional[ColumnConfig] = None):
        self.cfg = cfg or ColumnConfig()
        self.areas: Dict[str, TensorizedCorticalSheet] = {}
        self.ff_weights: Dict[Tuple[str, str], np.ndarray] = {}
        self.fb_weights: Dict[Tuple[str, str], np.ndarray] = {}
        self.t = 0
        self.v2_input_buffer = []
        
    def add_area(self, name: str, n_cols: int = 100):
        self.areas[name] = TensorizedCorticalSheet(n_cols=n_cols, cfg=self.cfg)
        
    def link_ff(self, src: str, dst: str):
        """Feedforward: src.h -> dst.x_bottom"""
        n_src = self.areas[src].n_cols
        n_dst = self.areas[dst].n_cols
        # FF projection: [n_dst, d_in_dst, d_h_src]
        # For simplicity, assume d_in_dst == d_h_src
        self.ff_weights[(src, dst)] = np.random.normal(0, 0.1, (n_dst, self.cfg.d_in, self.cfg.d_h))
        
    def link_fb(self, src: str, dst: str):
        """Feedback: src.h -> dst.x_top (apical)"""
        n_src = self.areas[src].n_cols
        n_dst = self.areas[dst].n_cols
        # FB projection: [n_dst, d_ctx_dst, d_h_src]
        self.fb_weights[(src, dst)] = np.random.normal(0, 0.1, (n_dst, self.cfg.d_ctx, self.cfg.d_h))

    def step(self, input_v1: np.ndarray, reward: float = 0.0) -> Dict:
        # ── 9.2 Predictive Feedback ──
        # V2 predicts V1 activity
        v1_prediction = np.zeros_like(input_v1)
        if ("V2", "V1") in self.fb_weights:
            h_v2_avg = np.mean(self.areas["V2"].h, axis=0)
            # Project d_h_v2 -> d_in_v1
            # We reuse fb_weights but need to match d_in_v1 shape
            # For 9.2 we'll add a specific prediction tensor
            v1_prediction = np.mean(np.einsum('cdh,h->cd', self.fb_weights[("V2", "V1")], h_v2_avg), axis=0)
            v1_prediction = v1_prediction[:len(input_v1)] # Truncate/pad to match

        # V1 sees (Actual - Predicted) = Residual
        v1_residual = input_v1 - 0.5 * v1_prediction
        
        # 1. Feedback context (V2 -> V1 apical)
        v1_apical = None
        if ("V2", "V1") in self.fb_weights:
            h_v2_avg = np.mean(self.areas["V2"].h, axis=0)
            v1_apical = np.einsum('cdh,h->cd', self.fb_weights[("V2", "V1")], h_v2_avg)

        # 2. Step V1 with residual
        res_v1 = self.areas["V1"].step(v1_residual, x_top=v1_apical, reward=reward)
        
        # 3. Feedforward (V1 -> V2)
        v1_h_avg = np.mean(self.areas["V1"].h, axis=0)
        v2_input = np.einsum('cdh,h->cd', self.ff_weights[("V1", "V2")], v1_h_avg)
        self.v2_input_buffer.append(np.mean(v2_input, axis=0))
        
        # 4. Step V2 (Temporal Pooling: every 5 steps)
        res_v2 = None
        if (self.t + 1) % 5 == 0:
            v2_input_pooled = np.mean(self.v2_input_buffer, axis=0)
            res_v2 = self.areas["V2"].step(v2_input_pooled, reward=reward)
            self.v2_input_buffer = []
        
        self.t += 1
        return {
            "v1": res_v1,
            "v2": res_v2,
            "v1_pred_err": np.linalg.norm(input_v1 - v1_prediction)
        }

if __name__ == "__main__":
    hier = CortexHierarchy()
    hier.add_area("V1", n_cols=50)
    hier.add_area("V2", n_cols=20)
    hier.link_ff("V1", "V2")
    hier.link_fb("V2", "V1")
    
    print("Multi-area Hierarchy initialized.")
    for i in range(10):
        out = hier.step(np.random.normal(0, 1, 16))
        print(f"Step {i} | V1 Error: {out['v1']['mean_e_mag']:.3f} | V2 Error: {out['v2']['mean_e_mag']:.3f}")
