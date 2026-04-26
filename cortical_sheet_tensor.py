"""
Cortical Microcircuit — Phase 8.1: Sensorimotor Integration

Spec v0.6 — Vectorized cortical sheet with motor layer (L5b).

Key upgrades:
- Tensorized Architecture (7.1-7.2)
- Structural & Multi-timescale Plasticity (7.3-7.4)
- Generative Sleep/Replay (7.5)
- Local Thalamic Scaling (7.6) — Neighborhood routing
- Motor Output Layer (8.1) — Dopamine-modulated action selection
"""

import numpy as np
from scipy.sparse import csr_matrix
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
import json

# Import Phase 0+1 foundation
from cortical_column import (
    ColumnConfig, NeuromodState, InhibitorySystem,
    _sigmoid, _tanh,
)
from engram_memory import EngramMemory

class TensorizedCorticalSheet:
    def __init__(self, n_cols: int = 100,
                 cfg: Optional[ColumnConfig] = None,
                 k_active: int = 4,
                 k_neighbors: int = 4,
                 rng_seed: int = 0):
        self.cfg    = cfg or ColumnConfig()
        self.n_cols = n_cols
        self.k      = k_active
        self.neighborhood_size = 25
        rng         = np.random.default_rng(rng_seed)

        scale_b = 1.0 / np.sqrt(self.cfg.d_in)
        scale_a = 1.0 / np.sqrt(self.cfg.d_ctx)
        scale_r = 0.1 / np.sqrt(self.cfg.d_h)
        scale_l = 0.05 / np.sqrt(self.cfg.d_lat)

        self.W_basal     = rng.normal(0, scale_b, (n_cols, self.cfg.d_h, self.cfg.d_in))
        self.W_pred_ff   = rng.normal(0, scale_b, (n_cols, self.cfg.d_in, self.cfg.d_h))
        self.W_apical    = rng.normal(0, scale_a, (n_cols, self.cfg.d_h, self.cfg.d_ctx))
        self.W_pred_ctx  = rng.normal(0, scale_b, (n_cols, self.cfg.d_in, self.cfg.d_h))
        self.W_recurrent = rng.normal(0, scale_r, (n_cols, self.cfg.d_h, self.cfg.d_h))
        self.W_h_to_ctx  = rng.normal(0, scale_r, (n_cols, self.cfg.d_ctx, self.cfg.d_h))
        self.W_lateral   = rng.normal(0, scale_l, (n_cols, self.cfg.d_lat, self.cfg.d_h))

        # ── 8.1 Motor Layer ──
        self.d_action = self.cfg.d_action
        self.W_motor  = rng.normal(0, 0.1, (n_cols, self.d_action, self.cfg.d_h))

        # ── Engram Memory (v1.0 Section 7.2) ──
        self.memory = EngramMemory(d_key=self.cfg.d_h, d_val=self.cfg.d_ctx, max_size=2000)

        self.W_basal_stable  = np.zeros_like(self.W_basal)
        self.W_apical_stable = np.zeros_like(self.W_apical)
        self.e_trace_b_slow  = np.zeros_like(self.W_basal)
        self.e_trace_a_slow  = np.zeros_like(self.W_apical)

        self.W_long_range = np.zeros((n_cols, n_cols))
        self.adjacency = self._build_sparse_adjacency(rng, k_neighbors)

        self.h = np.zeros((n_cols, self.cfg.d_h))
        self.p = np.ones(n_cols) * 0.3
        self.route_scores = np.ones(n_cols) * 0.5
        self._route_ema = np.ones(n_cols) * 0.5

        self.e_trace_b   = np.zeros((n_cols, self.cfg.d_h, self.cfg.d_in))
        self.e_trace_a   = np.zeros((n_cols, self.cfg.d_h, self.cfg.d_ctx))
        self.e_trace_ff  = np.zeros((n_cols, self.cfg.d_in, self.cfg.d_h))
        self.e_trace_ctx = np.zeros((n_cols, self.cfg.d_in, self.cfg.d_h))

        self.pv_gain  = np.ones(n_cols)
        self.sst_sup  = np.zeros(n_cols)
        self.vip_gate = np.zeros(n_cols)

        self.active_mask = np.zeros(n_cols, dtype=bool)
        self.neuromod     = NeuromodState()
        self.t            = 0
        self.p_structural = 0.01

    def _build_sparse_adjacency(self, rng, k_neighbors):
        rows, cols = [], []
        k_half = k_neighbors // 2
        for i in range(self.n_cols):
            for d in range(1, k_half + 1):
                j = (i + d) % self.n_cols
                rows.extend([i, j])
                cols.extend([j, i])
            j_far = rng.integers(0, self.n_cols)
            if j_far != i:
                rows.extend([i, j_far])
                cols.extend([j_far, i])
        data = np.ones(len(rows), dtype=bool)
        return csr_matrix((data, (rows, cols)), shape=(self.n_cols, self.n_cols))

    def _prune_and_regrow(self):
        mask = self.W_long_range > 0.05
        self.W_long_range *= mask
        rows, cols = [], []
        k_half = self.cfg.k_neighbors // 2
        for i in range(self.n_cols):
            for d in range(1, k_half + 1):
                j = (i + d) % self.n_cols
                rows.extend([i, j])
                cols.extend([j, i])
        w_rows, w_cols = np.where(mask)
        rows.extend(w_rows)
        cols.extend(w_cols)
        n_growth = int(self.n_cols * self.p_structural)
        for _ in range(n_growth):
            i, j = np.random.randint(0, self.n_cols, 2)
            if i != j:
                rows.extend([i, j])
                cols.extend([j, i])
        data = np.ones(len(rows), dtype=bool)
        self.adjacency = csr_matrix((data, (rows, cols)), shape=(self.n_cols, self.n_cols))

    def _gather_lateral(self, h_states, active_mask):
        h_active = h_states * active_mask[:, None]
        lateral_raw = self.adjacency @ h_active
        return np.einsum('cdh,ch->cd', self.W_lateral, lateral_raw)

    def _update_inhibition(self, soma_sparse, neuromod, err_mag):
        activity_level = np.mean(np.abs(soma_sparse), axis=1)
        self.pv_gain = np.clip(activity_level * neuromod.pv_gain_scale() + 0.5, *self.cfg.pv_clip)
        self.vip_gate = np.clip(neuromod.ne - 0.5, 0, 1)
        self.sst_sup = np.clip(err_mag * 0.3 * (1.0 - self.vip_gate * 0.8), *self.cfg.sst_clip)

    def _local_update_batch(self, W, trace, pre, post, e_mag, lr, decay, anti):
        hebb = np.einsum('co,ci->coi', post, pre)
        new_trace = decay * trace + (1 - decay) * hebb
        delta = lr * e_mag[:, None, None] * new_trace
        delta -= lr * anti * np.mean(post**2, axis=1)[:, None, None] * np.sign(hebb)
        W += delta
        norms = np.linalg.norm(W, axis=(1, 2), keepdims=True)
        W *= np.where(norms > 5.0, 5.0 / norms, 1.0)
        return new_trace, np.linalg.norm(delta, axis=(1, 2))

    def sleep_step(self, n_steps: int = 10):
        nm_sleep = NeuromodState(ach=0.1, da=0.5, ne=0.1)
        x_gen = _tanh(np.einsum('cdi,ch->cd', self.W_pred_ff, self.h))
        for _ in range(n_steps):
            x_input = np.mean(x_gen, axis=0)
            self.step(x_input, learn=True, neuromod_override=nm_sleep)
            x_gen = _tanh(np.einsum('cdi,ch->cd', self.W_pred_ff, self.h))
            self.W_basal_stable += 0.005 * self.e_trace_b_slow
            self.W_apical_stable += 0.005 * self.e_trace_a_slow
        self._prune_and_regrow()

    def step(self, x_bottom, x_top=None, reward=0.0, learn=True, neuromod_override=None):
        cfg = self.cfg
        self.t += 1
        nm = neuromod_override or self.neuromod
        if learn and self.t % 100 == 0: self._prune_and_regrow()

        noise = np.random.default_rng(self.t * 100).normal(0, 0.05, (self.n_cols, cfg.d_in))
        x_bottom_all = x_bottom[None, :] + noise
        if x_top is not None and x_top.ndim == 2:
            x_top_all = x_top
        else:
            x_top_all = np.tile(x_top if x_top is not None else np.zeros(cfg.d_ctx), (self.n_cols, 1))

        W_b_eff, W_a_eff = self.W_basal + self.W_basal_stable, self.W_apical + self.W_apical_stable

        # ── Memory Retrieval (Section 7.2) ──
        h_avg = np.mean(self.h, axis=0)
        mem_ctx, mem_conf = self.memory.retrieve(h_avg)

        basal = np.einsum('cdi,ci->cd', W_b_eff, x_bottom_all)
        x_top_aug = x_top_all + 0.3 * np.einsum('cdh,ch->cd', self.W_h_to_ctx, self.h)
        
        # Integrate memory context into apical drive
        x_top_aug += 0.4 * mem_conf * mem_ctx[None, :]

        apical_gate = _sigmoid(np.einsum('cdi,ci->cd', W_a_eff, x_top_aug) * (1.0 - self.sst_sup[:, None]))
        
        soma = basal * apical_gate
        lat_drive = self._gather_lateral(self.h, self.active_mask)
        lat_h = np.pad(lat_drive, ((0,0), (0, cfg.d_h - lat_drive.shape[1]))) if lat_drive.shape[1] < cfg.d_h else lat_drive[:, :cfg.d_h]
        
        soma_gated = (soma + 0.2 * lat_h) / self.pv_gain[:, None]
        thresholds = np.percentile(np.abs(soma_gated), 75, axis=1, keepdims=True)
        soma_sparse = soma_gated * (np.abs(soma_gated) >= thresholds).astype(float)

        y_pred_ff = _tanh(np.einsum('cdi,ch->cd', self.W_pred_ff, self.h))
        y_pred_ctx = _tanh(np.einsum('cdi,ch->cd', self.W_pred_ctx, self.h * apical_gate))
        p_w = np.clip(self.p, 0, 1)[:, None]
        y_pred = (1 - p_w) * y_pred_ff + p_w * y_pred_ctx

        e_ff, e_ctx = x_bottom_all - y_pred_ff, x_bottom_all - y_pred_ctx
        err_mag = np.linalg.norm(x_bottom_all - y_pred, axis=1)

        self.p = 0.95 * self.p + 0.05 * (1.0 / (err_mag + 1e-4))
        self._update_inhibition(soma_sparse, nm, err_mag)

        if learn and not nm.exploration_mode():
            decay, lr, anti = np.exp(-1.0 / cfg.tau_elig), cfg.lr_base * nm.learning_rate_scale(), cfg.anti_hebb_scale
            self.e_trace_b, _ = self._local_update_batch(self.W_basal, self.e_trace_b, x_bottom_all, soma_sparse, err_mag, lr, decay, anti)
            self.e_trace_a, _ = self._local_update_batch(self.W_apical, self.e_trace_a, x_top_aug, soma_sparse, err_mag, lr, decay, anti)
            self.e_trace_ff, _ = self._local_update_batch(self.W_pred_ff, self.e_trace_ff, self.h, e_ff, err_mag, lr, decay, anti)
            self.e_trace_ctx, _ = self._local_update_batch(self.W_pred_ctx, self.e_trace_ctx, self.h, e_ctx, err_mag, lr, decay, anti)
            
            dec_s = np.exp(-1.0/500.0)
            self.e_trace_b_slow = dec_s * self.e_trace_b_slow + (1-dec_s) * np.einsum('co,ci->coi', soma_sparse, x_bottom_all)
            self.e_trace_a_slow = dec_s * self.e_trace_a_slow + (1-dec_s) * np.einsum('co,ci->coi', soma_sparse, x_top_aug)
            b_tag, a_tag = self.e_trace_b_slow > 0.1, self.e_trace_a_slow > 0.1
            self.W_basal_stable[b_tag] += 0.001 * self.W_basal[b_tag]
            self.W_apical_stable[a_tag] += 0.001 * self.W_apical[a_tag]

        self.h = _tanh(soma_sparse + 0.5 * np.einsum('cdh,ch->cd', self.W_recurrent, self.h))
        self._route_ema = (1 - 1.0/cfg.tau_route) * self._route_ema + (1.0/cfg.tau_route) * (self.p * np.mean(np.abs(soma_sparse), axis=1))
        self.route_scores = np.clip(self._route_ema, 0, 1)
        self.active_mask = self._route_active()

        if learn: self._update_long_range()
        if neuromod_override is None: self._update_neuromod(err_mag, reward)

        # ── Memory update (store trace, Section 7.2) ──
        if learn and nm.da > 1.2:  # High DA reinforces memory
            h_active_avg = np.mean(self.h[self.active_mask], axis=0) if np.any(self.active_mask) else h_avg
            self.memory.store(h_active_avg, np.mean(x_top_aug, axis=0), strength=nm.da)
        self.memory.decay()

        # ── 8.1 Motor Generation ──
        proposals = np.einsum('cah,ch->ca', self.W_motor, self.h)
        weighted = proposals * (self.active_mask[:, None] * self.route_scores[:, None])
        
        # ── 8.2 NE-driven Exploration ──
        explore_noise = np.random.normal(0, nm.ne * 0.2, self.d_action)
        action = np.tanh(np.sum(weighted, axis=0) + explore_noise)

        if learn:
            da_mod = nm.da - 1.0
            self.W_motor += 0.01 * da_mod * np.einsum('ca,ch->cah', weighted, self.h)
            m_norms = np.linalg.norm(self.W_motor, axis=(1,2), keepdims=True)
            self.W_motor *= np.where(m_norms > 1.0, 1.0 / m_norms, 1.0)

        return {"mean_e_mag": float(np.mean(err_mag)), "action": action}

    def _route_active(self):
        n_g = max(1, self.n_cols // self.neighborhood_size)
        k_g = max(1, self.k // n_g)
        new_active = np.zeros(self.n_cols, dtype=bool)
        for g in range(n_g):
            start = g * self.neighborhood_size
            end = min(start + self.neighborhood_size, self.n_cols)
            s = self.route_scores[start:end]
            if len(s) == 0: continue
            idx = np.argsort(s)[::-1]
            thr = s[idx[min(k_g, len(s)) - 1]]
            new_active[start:end] = (s >= thr)
        return new_active | (self.active_mask & (self.route_scores >= 0.1))

    def _update_long_range(self):
        act = np.where(self.active_mask)[0]
        if len(act) < 2: return
        i, j = np.meshgrid(act, act, indexing='ij')
        m = i != j
        if1, jf1 = i[m], j[m]
        self.W_long_range[if1, jf1] += self.cfg.lr_base * 0.01 * (self.route_scores[if1] * self.route_scores[jf1] - 0.01 * self.W_long_range[if1, jf1])

    def _update_neuromod(self, err_mag, reward):
        me = float(np.mean(err_mag))
        self.neuromod.ach = 0.67 * self.neuromod.ach + 0.33 * np.clip(0.5 + me * 0.5, 0.3, 2.5)
        self.neuromod.da = 0.93 * self.neuromod.da + 0.07 * np.clip(1.0 + reward, 0.1, 2.5)
        self.neuromod.ne = 0.99 * self.neuromod.ne + 0.01 * np.clip(me * 0.4, 0.1, 1.5)

if __name__ == "__main__":
    print("Cortical Sheet Tensor Module Loaded.")
    print("Run benchmark.py or bench_nav.py for tests.")
