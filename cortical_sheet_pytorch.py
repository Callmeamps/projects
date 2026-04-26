"""
Cortical Sheet — PyTorch GPU Version (Phase 10.2)

Port of TensorizedCorticalSheet to PyTorch for GPU acceleration.
Supports CUDA if available.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
from cortical_column import ColumnConfig, NeuromodState, _sigmoid, _tanh
from engram_memory import EngramMemory


class PyTorchedCorticalSheet:
    """PyTorch version of TensorizedCorticalSheet for GPU acceleration."""

    def __init__(self, n_cols: int = 100,
                 cfg: Optional[ColumnConfig] = None,
                 k_active: int = 4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.cfg = cfg or ColumnConfig()
        self.n_cols = n_cols
        self.k = k_active
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Initialize weights as torch tensors
        scale_b = 1.0 / np.sqrt(self.cfg.d_in)
        scale_a = 1.0 / np.sqrt(self.cfg.d_ctx)
        scale_r = 0.1 / np.sqrt(self.cfg.d_h)
        scale_l = 0.05 / np.sqrt(self.cfg.d_lat)

        # Use torch.randn with manual seed for reproducibility
        generator = torch.Generator(device=self.device)
        generator.manual_seed(0)

        self.W_basal = torch.randn(n_cols, self.cfg.d_h, self.cfg.d_in,
                                   generator=generator, device=self.device) * scale_b
        self.W_pred_ff = torch.randn(n_cols, self.cfg.d_in, self.cfg.d_h,
                                     generator=generator, device=self.device) * scale_b
        self.W_apical = torch.randn(n_cols, self.cfg.d_h, self.cfg.d_ctx,
                                    generator=generator, device=self.device) * scale_a
        self.W_pred_ctx = torch.randn(n_cols, self.cfg.d_in, self.cfg.d_h,
                                      generator=generator, device=self.device) * scale_b
        self.W_recurrent = torch.randn(n_cols, self.cfg.d_h, self.cfg.d_h,
                                      generator=generator, device=self.device) * scale_r
        self.W_h_to_ctx = torch.randn(n_cols, self.cfg.d_ctx, self.cfg.d_h,
                                      generator=generator, device=self.device) * scale_r
        self.W_lateral = torch.randn(n_cols, self.cfg.d_lat, self.cfg.d_h,
                                     generator=generator, device=self.device) * scale_l

        # Motor layer
        self.d_action = 4
        self.W_motor = torch.randn(n_cols, self.d_action, self.cfg.d_h,
                                   generator=generator, device=self.device) * 0.1

        # Engram memory (keep numpy for now, will convert on retrieval)
        self.memory = EngramMemory(d_key=self.cfg.d_h, d_val=self.cfg.d_ctx, max_size=2000)

        # State tensors
        self.h = torch.zeros(n_cols, self.cfg.d_h, device=self.device)
        self.p = torch.ones(n_cols, device=self.device) * 0.3
        self.route_scores = torch.ones(n_cols, device=self.device) * 0.5
        self._route_ema = torch.ones(n_cols, device=self.device) * 0.5

        # Inhibition
        self.pv_gain = torch.ones(n_cols, device=self.device)
        self.sst_sup = torch.zeros(n_cols, device=self.device)
        self.vip_gate = torch.zeros(n_cols, device=self.device)

        self.active_mask = torch.zeros(n_cols, dtype=torch.bool, device=self.device)
        self.neuromod = NeuromodState()
        self.t = 0

        # Make weights require gradients? No - we update manually
        for param in [self.W_basal, self.W_pred_ff, self.W_apical,
                     self.W_pred_ctx, self.W_recurrent, self.W_h_to_ctx,
                     self.W_lateral, self.W_motor]:
            param.requires_grad = False

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor on device."""
        return torch.from_numpy(arr).to(self.device) if isinstance(arr, np.ndarray) else arr

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy."""
        return t.detach().cpu().numpy()

    def step(self, x_bottom, x_top=None, reward=0.0, learn=True):
        """Single timestep (PyTorch version)."""
        cfg = self.cfg
        self.t += 1

        # Convert inputs to tensors
        if isinstance(x_bottom, np.ndarray):
            x_bottom = self._to_tensor(x_bottom)
        x_bottom_all = x_bottom.unsqueeze(0) if x_bottom.dim() == 1 else x_bottom

        if x_top is not None:
            if isinstance(x_top, np.ndarray):
                x_top = self._to_tensor(x_top)
            x_top_all = x_top.unsqueeze(0) if x_top.dim() == 1 else x_top
        else:
            x_top_all = torch.zeros(self.n_cols, cfg.d_ctx, device=self.device)

        # Basal forward
        basal = torch.einsum('cdi,ci->cd', self.W_basal, x_bottom_all)

        # Apical with context
        x_top_aug = x_top_all + 0.3 * torch.einsum('cdh,ch->cd', self.W_h_to_ctx, self.h)

        # Memory retrieval (convert to numpy, then back)
        h_avg = torch.mean(self.h, dim=0).detach().cpu().numpy()
        mem_ctx, mem_conf = self.memory.retrieve(h_avg)
        mem_ctx_tensor = self._to_tensor(mem_ctx)
        mem_conf_tensor = torch.tensor(mem_conf, device=self.device)

        x_top_aug += 0.4 * mem_conf_tensor * mem_ctx_tensor.unsqueeze(0)

        apical_gate = torch.sigmoid(
            torch.einsum('cdi,ci->cd', self.W_apical, x_top_aug) * (1.0 - self.sst_sup).unsqueeze(1)
        )

        soma = basal * apical_gate
        soma_gated = soma / self.pv_gain.unsqueeze(1)

        # Sparsity threshold
        thresholds = torch.quantile(torch.abs(soma_gated), 0.75, dim=1, keepdim=True)
        soma_sparse = soma_gated * (torch.abs(soma_gated) >= thresholds).float()

        # Prediction
        y_pred_ff = torch.tanh(torch.einsum('cdi,ch->cd', self.W_pred_ff, self.h))
        y_pred_ctx = torch.tanh(torch.einsum('cdi,ch->cd', self.W_pred_ctx, self.h * apical_gate))
        p_w = torch.clamp(self.p, 0, 1).unsqueeze(1)
        y_pred = (1 - p_w) * y_pred_ff + p_w * y_pred_ctx

        err_mag = torch.norm(x_bottom_all - y_pred, dim=1)

        # Update precision
        self.p = 0.95 * self.p + 0.05 * (1.0 / (err_mag + 1e-4))

        # Update inhibition
        activity_level = torch.mean(torch.abs(soma_sparse), dim=1)
        self.pv_gain = torch.clamp(activity_level * 1.0 + 0.5, *cfg.pv_clip)
        self.vip_gate = torch.clamp(torch.tensor(self.neuromod.ne - 0.5), 0, 1)
        self.sst_sup = torch.clamp(err_mag * 0.3 * (1.0 - self.vip_gate), *cfg.sst_clip)

        # Update state
        rec = 0.5 * torch.einsum('cdh,ch->cd', self.W_recurrent, self.h)
        self.h = torch.tanh(soma_sparse + rec)

        # Routing
        self._route_ema = (1 - 1.0/cfg.tau_route) * self._route_ema + (1.0/cfg.tau_route) * \
                          (self.p * torch.mean(torch.abs(soma_sparse), dim=1))
        self.route_scores = torch.clamp(self._route_ema, 0, 1)

        # Top-k active
        k = max(1, int(self.n_cols * 0.05))
        topk_vals, topk_idx = torch.topk(self.route_scores, k)
        self.active_mask = torch.zeros(self.n_cols, dtype=torch.bool, device=self.device)
        self.active_mask[topk_idx] = True

        # Motor output
        proposals = torch.einsum('cah,ch->ca', self.W_motor, self.h)
        weighted = proposals * (self.active_mask.float().unsqueeze(1) * self.route_scores.unsqueeze(1))

        action = torch.tanh(torch.sum(weighted, dim=0))

        # Memory update
        if learn and self.neuromod.da > 1.2:
            h_active = self.h[self.active_mask] if self.active_mask.any() else self.h
            h_active_avg = torch.mean(h_active, dim=0).detach().cpu().numpy()
            x_top_np = x_top_aug.detach().cpu().numpy()
            self.memory.store(h_active_avg, np.mean(x_top_np, axis=0), strength=self.neuromod.da)
        self.memory.decay()

        # Update neuromod
        if learn:
            me = float(torch.mean(err_mag))
            self.neuromod.ach = 0.67 * self.neuromod.ach + 0.33 * np.clip(0.5 + me * 0.5, 0.3, 2.5)
            self.neuromod.da = 0.93 * self.neuromod.da + 0.07 * np.clip(1.0 + reward, 0.1, 2.5)
            self.neuromod.ne = 0.99 * self.neuromod.ne + 0.01 * np.clip(me * 0.4, 0.1, 1.5)

        return {
            "mean_e_mag": float(torch.mean(err_mag)),
            "action": self._to_numpy(action),
            "h": self._to_numpy(self.h),
        }

    def benchmark(self, steps: int = 100):
        """Quick benchmark comparing to numpy version."""
        import time

        x = torch.randn(self.cfg.d_in, device=self.device)

        start = time.time()
        for _ in range(steps):
            self.step(x, learn=False)
        elapsed = time.time() - start

        print(f"PyTorch ({self.device}): {steps} steps in {elapsed:.3f}s ({steps/elapsed:.1f} steps/s)")
        return elapsed


if __name__ == "__main__":
    print("PyTorch Cortical Sheet")
    sheet = PyTorchedCorticalSheet(n_cols=100, cfg=ColumnConfig(d_in=16, d_h=32))
    sheet.benchmark(100)
