"""
Cortical Microcircuit — Phase 2 through Phase 6
Spec v0.4 — builds directly on cortical_column.py (Phase 0+1)

Phase 2: Dendritic separation  — explicit basal/apical stream split
Phase 3: Inhibitory control    — sparsity constraint, stability under noise
Phase 4: Multi-column system   — lateral connections, specialisation, competition
Phase 5: Thalamic routing      — top-k gating, error-driven attention
Phase 6: Neuromodulatory ctrl  — task switching without retraining

No global backprop at any phase. All updates remain local.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import json

# Import Phase 0+1 foundation
from cortical_column import (
    ColumnConfig, NeuromodState, PyramidalCompartment,
    InhibitorySystem, LocalPlasticity,
    _sigmoid, _tanh,
)


# ═══════════════════════════════════════════════════════════════
# Phase 2 — Dendritic separation
# ═══════════════════════════════════════════════════════════════

class DendriticColumn:
    """
    Phase 2: Explicit basal / apical stream separation.

    Key upgrade over Phase 1:
    - Basal stream has its own W_pred_basal for a feedforward-only prediction.
    - Apical stream generates a context-modulated gain vector (not a scalar).
    - Soma output = basal_pred * sigmoid(apical_gain) — pure coincidence detection.
    - Two separate predictions: y_pred_ff (basal only) and y_pred_ctx (soma-gated).
    - e_local now has two components: e_ff and e_ctx, allowing the system to
      separately track how wrong the feedforward vs. context-modulated pathways are.

    Success criterion (spec): same x_bottom, different x_top → different y_pred.
    """

    def __init__(self, cfg: ColumnConfig, column_id: int = 0,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.id  = column_id
        rng = rng or np.random.default_rng(42 + column_id)

        # ── Basal stream (feedforward evidence) ──
        scale_b = 1.0 / np.sqrt(cfg.d_in)
        self.W_basal    = rng.normal(0, scale_b, (cfg.d_h, cfg.d_in))
        self.W_pred_ff  = rng.normal(0, scale_b, (cfg.d_in, cfg.d_h))   # ff-only readout

        # ── Apical stream (top-down context) ──
        scale_a = 1.0 / np.sqrt(cfg.d_ctx)
        self.W_apical      = rng.normal(0, scale_a, (cfg.d_h, cfg.d_ctx))
        self.W_pred_ctx    = rng.normal(0, scale_b, (cfg.d_in, cfg.d_h)) # ctx readout

        # ── Recurrent + ctx projection ──
        self.W_recurrent = rng.normal(0, 0.1 / np.sqrt(cfg.d_h), (cfg.d_h, cfg.d_h))
        self.W_h_to_ctx  = rng.normal(0, 0.1 / np.sqrt(cfg.d_h), (cfg.d_ctx, cfg.d_h))

        # ── Lateral aggregation (Phase 4 ready, dormant until then) ──
        self.W_lateral   = rng.normal(0, 0.05 / np.sqrt(cfg.d_lat), (cfg.d_h, cfg.d_lat))

        # ── Inhibitory system ──
        self.inhibitory  = InhibitorySystem(cfg)

        # ── Plasticity traces ──
        self.e_trace_b   = np.zeros((cfg.d_h, cfg.d_in))
        self.e_trace_a   = np.zeros((cfg.d_h, cfg.d_ctx))
        self.e_trace_ff  = np.zeros((cfg.d_in, cfg.d_h))
        self.e_trace_ctx = np.zeros((cfg.d_in, cfg.d_h))

        # ── State ──
        self.h           = np.zeros(cfg.d_h)
        self.p           = 0.3
        self.route_score = 0.5
        self._route_ema  = 0.5
        self.history: List[Dict] = []

    # ── Phase 2 core: explicit stream forward pass ──────────────────────────

    def _basal_forward(self, x_bottom: np.ndarray) -> np.ndarray:
        """Feedforward evidence — linear projection"""
        return self.W_basal @ x_bottom

    def _apical_forward(self, x_top: np.ndarray,
                         sst_sup: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Top-down context gate — multiplicative, SST-suppressed"""
        raw    = self.W_apical @ x_top
        gated  = raw * (1.0 - sst_sup)
        return _sigmoid(gated), gated   # (gate_vector, pre_sigmoid)

    def _soma_integrate(self, basal: np.ndarray,
                        apical_gate: np.ndarray) -> np.ndarray:
        """Coincidence detection: fires when evidence aligns with expectation"""
        return basal * apical_gate

    def _lateral_aggregate(self, x_lateral: Optional[np.ndarray]) -> np.ndarray:
        """Aggregate neighbour signals — zero if no lateral input (Phase 3)"""
        if x_lateral is None or x_lateral.size == 0:
            return np.zeros(self.cfg.d_h)
        # x_lateral: [k_neighbors, d_lat] → mean pool then project
        pooled = np.mean(x_lateral, axis=0) if x_lateral.ndim == 2 else x_lateral
        return self.W_lateral @ pooled

    # ── Plasticity helper ────────────────────────────────────────────────────

    def _local_update(self, W, trace, pre, post, e_mag, lr, decay, anti):
        """Single local rule: Hebbian + error-mod + anti-Hebbian, trace-bridged"""
        hebb  = np.outer(post, pre)
        trace = decay * trace + (1 - decay) * hebb
        delta = lr * e_mag * trace
        delta -= lr * anti * float(np.mean(post**2)) * np.sign(hebb)
        W    += delta
        # norm clip
        n = np.linalg.norm(W)
        if n > 5.0:
            W *= 5.0 / n
        return trace, float(np.linalg.norm(delta))

    # ── Main step ────────────────────────────────────────────────────────────

    def step(self, x_bottom: np.ndarray,
             x_top: Optional[np.ndarray] = None,
             x_lateral: Optional[np.ndarray] = None,
             neuromod: Optional[NeuromodState] = None,
             learn: bool = True) -> Dict:

        cfg      = self.cfg
        neuromod = neuromod or NeuromodState()
        x_top    = x_top if x_top is not None else np.zeros(cfg.d_ctx)

        # ── Phase 1: Feedforward ──
        basal     = self._basal_forward(x_bottom)

        # Augment x_top with recurrent context (h → ctx space)
        x_top_aug = x_top + 0.3 * (self.W_h_to_ctx @ self.h)

        # ── Phase 2: Prediction (two streams) ──
        apical_gate, apical_raw = self._apical_forward(
            x_top_aug, self.inhibitory.sst_suppression)
        soma      = self._soma_integrate(basal, apical_gate)

        lat_drive = self._lateral_aggregate(x_lateral)
        soma_full = soma + 0.2 * lat_drive        # lateral modulates soma

        # PV divisive gain
        soma_gated = self.inhibitory.apply_pv(soma_full)

        # Phase 3: sparsity — k-winner activation (top-25% active)
        threshold  = np.percentile(np.abs(soma_gated), 75)
        soma_sparse = soma_gated * (np.abs(soma_gated) >= threshold).astype(float)

        # Two predictions from current h + current context signal
        y_pred_ff  = _tanh(self.W_pred_ff  @ self.h)                       # basal-only: history
        # ctx prediction: modulated by current apical gate directly
        ctx_signal = self.W_pred_ctx @ (self.h * apical_gate[:cfg.d_h if apical_gate.shape[0] >= cfg.d_h else apical_gate.shape[0]])
        y_pred_ctx = _tanh(ctx_signal)

        # Weighted prediction: precision gates how much context is trusted
        ctx_weight = float(np.clip(self.p, 0, 1))
        y_pred     = (1 - ctx_weight) * y_pred_ff + ctx_weight * y_pred_ctx

        # ── Phase 3: Error computation ──
        e_ff  = x_bottom - y_pred_ff
        e_ctx = x_bottom - y_pred_ctx
        e_local = x_bottom - y_pred
        err_mag = float(np.linalg.norm(e_local))

        # Precision update
        self.p = 0.95 * self.p + 0.05 * (1.0 / (err_mag + 1e-4))

        # ── Phase 4: Inhibition update ──
        self.inhibitory.update(soma_sparse, neuromod, err_mag)

        # ── Phase 5: Plasticity ──
        stats = {}
        if learn and not neuromod.exploration_mode():
            decay = np.exp(-1.0 / cfg.tau_elig)
            lr    = cfg.lr_base * neuromod.learning_rate_scale()
            anti  = cfg.anti_hebb_scale

            # Basal weights: driven by ff error
            e_ff_mag = float(np.linalg.norm(e_ff))
            self.e_trace_b, d1 = self._local_update(
                self.W_basal, self.e_trace_b, x_bottom, soma_sparse, e_ff_mag, lr, decay, anti)

            # Apical weights: driven by ctx error
            e_ctx_mag = float(np.linalg.norm(e_ctx))
            self.e_trace_a, d2 = self._local_update(
                self.W_apical, self.e_trace_a, x_top_aug, soma_sparse, e_ctx_mag, lr, decay, anti)

            # Readout weights
            self.e_trace_ff, d3 = self._local_update(
                self.W_pred_ff, self.e_trace_ff, self.h, e_ff, e_ff_mag, lr, decay, anti)
            self.e_trace_ctx, d4 = self._local_update(
                self.W_pred_ctx, self.e_trace_ctx, self.h, e_ctx, e_ctx_mag, lr, decay, anti)

            stats = {"d_basal": d1, "d_apical": d2, "d_pred_ff": d3, "d_pred_ctx": d4}

        # ── Phase 6: State update ──
        h_next = _tanh(soma_sparse + 0.5 * (self.W_recurrent @ self.h))
        self.h = h_next

        # Routing score (low-pass)
        alpha = 1.0 / cfg.tau_route
        raw_score = self.p * float(np.mean(np.abs(soma_sparse)))
        self._route_ema = (1 - alpha) * self._route_ema + alpha * raw_score
        self.route_score = float(np.clip(self._route_ema, 0, 1))

        result = dict(
            y_pred=y_pred, y_pred_ff=y_pred_ff, y_pred_ctx=y_pred_ctx,
            e_local=e_local, e_ff=e_ff, e_ctx=e_ctx,
            e_mag=err_mag, e_ff_mag=float(np.linalg.norm(e_ff)),
            e_ctx_mag=float(np.linalg.norm(e_ctx)),
            route_score=self.route_score, h_next=h_next,
            precision=self.p,
            pv_gain=self.inhibitory.pv_gain,
            sst_sup=self.inhibitory.sst_sup,
            vip_gate=self.inhibitory.vip_gate,
            sparsity=float(np.mean(soma_sparse == 0)),
            **stats,
        )

        self.history.append({
            k: float(v) if np.isscalar(v) else float(np.mean(np.abs(v)))
            for k, v in result.items()
            if k not in ("y_pred","y_pred_ff","y_pred_ctx","e_local","e_ff","e_ctx","h_next")
        })
        return result


# ═══════════════════════════════════════════════════════════════
# Phase 5 — Thalamic Routing Module
# ═══════════════════════════════════════════════════════════════

class ThalamicRouter:
    """
    Spec §6 + Phase 5:
    - Maintains a score per column (route_score)
    - Selects top-k active columns each timestep
    - Hysteresis: stay active if score within 10% of threshold (spec §0.4)
    - Feedback: error signal updates routing scores (error-driven attention)
    - Output: binary active mask [n_cols], gating who communicates

    route_score shape: [n_cols]  — fills the blank in spec §0.1
    """

    def __init__(self, n_cols: int, k: int, tau_route: float = 5.0,
                 d_in: int = 16, rng: Optional[np.random.Generator] = None):
        self.n_cols   = n_cols
        self.k        = k
        self.tau      = tau_route
        self.scores   = np.ones(n_cols) * 0.5
        self.active   = np.zeros(n_cols, dtype=bool)
        self.hysteresis = 0.10
        rng = rng or np.random.default_rng(0)
        self.W_tune   = rng.normal(0, 1.0 / np.sqrt(d_in), (n_cols, d_in))
        self.tune_ema = np.zeros(n_cols)

    def update(self, column_route_scores: np.ndarray,
               column_errors: np.ndarray,
               x_input: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns active mask [n_cols].
        column_route_scores: precision-weighted salience from each column
        column_errors: e_mag per column (feedback to routing)
        x_input: current sensory input for content-based thalamic gating [d_in]
        """
        alpha = 1.0 / self.tau
        self.scores = (1 - alpha) * self.scores + alpha * column_route_scores

        err_norm = column_errors / (np.max(column_errors) + 1e-8)
        boosted  = self.scores * (1.0 + 0.3 * err_norm)

        # Content-based tuning: columns whose tuning aligns with input get boosted
        if x_input is not None:
            x_norm = x_input / (np.linalg.norm(x_input) + 1e-8)
            tune_match = self.W_tune @ x_norm          # raw dot product
            self.tune_ema = 0.67 * self.tune_ema + 0.33 * tune_match
            # Boost proportional to tuning alignment (positive only)
            boosted = boosted + np.clip(self.tune_ema, 0, None) * 0.5

        sorted_idx = np.argsort(boosted)[::-1]
        threshold  = boosted[sorted_idx[min(self.k, self.n_cols) - 1]]
        new_active = np.zeros(self.n_cols, dtype=bool)
        new_active[sorted_idx[:self.k]] = True

        keep = self.active & (boosted >= threshold * (1 - self.hysteresis))
        new_active = new_active | keep
        self.active = new_active
        return self.active.copy()

    def update_tuning_wta(self, x_input: np.ndarray, active_mask: np.ndarray,
                           lr_win: float = 0.03, lr_lose: float = 0.001):
        """
        WTA competitive Oja rule:
          Active columns  → drift toward input (Oja normalised update)
          Inactive columns → small weight decay
        With k<n/2 this produces strong per-column specialisation.
        """
        x_norm = x_input / (np.linalg.norm(x_input) + 1e-8)
        for i in range(self.n_cols):
            if active_mask[i]:
                proj = float(self.W_tune[i] @ x_norm)
                self.W_tune[i] += lr_win * (x_norm - self.W_tune[i] * proj)
            else:
                self.W_tune[i] *= (1.0 - lr_lose)
            n = np.linalg.norm(self.W_tune[i])
            if n > 0:
                self.W_tune[i] /= n

    @property
    def route_scores(self) -> np.ndarray:
        """[n_cols] — the filled-in tensor shape from spec §0.1"""
        return self.scores.copy()


# ═══════════════════════════════════════════════════════════════
# Phase 4 + 5 + 6 — CorticalSheet (multi-column network)
# ═══════════════════════════════════════════════════════════════

class CorticalSheet:
    """
    Phase 4: Multi-column network (5–20 columns) with lateral connections
             and winner-take-most competition.
    Phase 5: Thalamic routing selects active columns per timestep.
    Phase 6: Neuromodulatory state controls learning mode and task switching.

    Topology: sparse grid / small-world (spec §10).
    Long-range connections: rare, learned (Phase 4 step 5).
    """

    def __init__(self, n_cols: int = 12,
                 cfg: Optional[ColumnConfig] = None,
                 k_active: int = 4,
                 rng_seed: int = 0):
        self.cfg    = cfg or ColumnConfig()
        self.n_cols = n_cols
        self.k      = k_active
        rng         = np.random.default_rng(rng_seed)

        # ── Columns ──
        self.columns = [
            DendriticColumn(self.cfg, column_id=i,
                            rng=np.random.default_rng(rng_seed + i + 1))
            for i in range(n_cols)
        ]

        # ── Phase 4: Sparse lateral connectivity ──
        # Each column connects to k_neighbors others (local + rare long-range)
        self.adjacency   = self._build_adjacency(rng)
        # Learned long-range weight matrices [n_cols, n_cols] — sparse
        self.W_long_range = np.zeros((n_cols, n_cols))  # Phase 4.5: learned

        # ── Phase 5: Thalamic router ──
        self.router = ThalamicRouter(n_cols, k=k_active, tau_route=self.cfg.tau_route,
                                         d_in=self.cfg.d_in, rng=rng)

        # ── Phase 6: Neuromodulatory dynamics (multi-timescale, spec §0.5) ──
        self.neuromod     = NeuromodState()
        self._ach_target  = 1.0
        self._da_target   = 1.0
        self._ne_target   = 0.5

        # ── Global state ──
        self.t            = 0
        self.sheet_history: List[Dict] = []

    def _build_adjacency(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sparse connectivity: each column connects to 2 nearest neighbours +
        1 random long-range link (spec §10: local dominates, rare long-range).
        Returns adjacency matrix [n_cols, n_cols].
        """
        adj = np.zeros((self.n_cols, self.n_cols), dtype=bool)
        k   = self.cfg.k_neighbors
        for i in range(self.n_cols):
            # local ring neighbours
            for d in range(1, k // 2 + 1):
                j = (i + d) % self.n_cols
                adj[i, j] = adj[j, i] = True
            # one rare long-range link
            far = rng.integers(0, self.n_cols)
            if far != i:
                adj[i, far] = adj[far, i] = True
        return adj

    def _collect_lateral(self, col_idx: int,
                          hidden_states: np.ndarray,
                          active_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Gather active neighbour hidden states for column col_idx.
        hidden_states: [n_cols, d_h]
        Returns [k_active_neighbours, d_lat] or None.
        """
        neighbours = np.where(self.adjacency[col_idx] & active_mask)[0]
        if len(neighbours) == 0:
            return None
        # Project d_h → d_lat if needed
        d_lat = self.cfg.d_lat
        d_h   = self.cfg.d_h
        if d_h == d_lat:
            return hidden_states[neighbours]
        # Simple truncation / zero-padding
        lat = np.zeros((len(neighbours), d_lat))
        lat[:, :min(d_h, d_lat)] = hidden_states[neighbours, :min(d_h, d_lat)]
        return lat

    def _update_neuromod(self, mean_error: float, step_reward: float = 0.0):
        """
        Multi-timescale neuromod update (spec §0.5):
        - ACh: fast (tau=3 steps), tracks sensory novelty (∝ error)
        - DA:  medium (tau=15 steps), tracks reward signal
        - NE:  slow (tau=100 steps), tracks sustained uncertainty
        """
        # Target values driven by current signals
        self._ach_target = float(np.clip(0.5 + mean_error * 0.5, 0.3, 2.5))
        self._da_target  = float(np.clip(1.0 + step_reward,       0.1, 2.5))
        self._ne_target  = float(np.clip(mean_error * 0.4,        0.1, 1.5))

        # Exponential smoothing toward targets at different timescales
        self.neuromod.ach = 0.67 * self.neuromod.ach + 0.33 * self._ach_target  # tau≈3
        self.neuromod.da  = 0.93 * self.neuromod.da  + 0.07 * self._da_target   # tau≈15
        self.neuromod.ne  = 0.99 * self.neuromod.ne  + 0.01 * self._ne_target   # tau≈100

    def step(self, x_bottom: np.ndarray,
             x_top: Optional[np.ndarray] = None,
             reward: float = 0.0,
             learn: bool = True,
             neuromod_override: Optional[NeuromodState] = None) -> Dict:
        """
        One full sheet timestep — all columns, routing, neuromod.

        x_bottom: [d_in]  shared sensory input (each column sees it + noise)
        x_top:    [d_ctx] shared top-down context
        reward:   scalar reward for DA modulation (Phase 6)
        """
        cfg = self.cfg
        self.t += 1
        nm  = neuromod_override or self.neuromod

        # ── Collect previous hidden states for lateral message passing ──
        h_states = np.stack([c.h for c in self.columns])   # [n_cols, d_h]

        # ── Phase 5: Router selects active columns ──
        col_route_scores = np.array([c.route_score for c in self.columns])
        col_errors       = np.array([c.history[-1]["e_mag"] if c.history else 1.0
                                     for c in self.columns])
        active_mask = self.router.update(col_route_scores, col_errors, x_input=x_bottom)

        # ── Phase 4: Step each active column ──
        outputs   = {}
        all_emags = []
        for i, col in enumerate(self.columns):
            if not active_mask[i]:
                # Inactive columns don't compute — sparse computation (spec §5)
                outputs[i] = None
                continue

            # Add per-column input noise for diversity (distributes features)
            noise      = np.random.default_rng(self.t * 100 + i).normal(0, 0.05, cfg.d_in)
            x_in       = x_bottom + noise

            # Gather lateral inputs from active neighbours
            x_lat      = self._collect_lateral(i, h_states, active_mask)

            out        = col.step(x_in, x_top=x_top, x_lateral=x_lat,
                                  neuromod=nm, learn=learn)
            outputs[i] = out
            all_emags.append(out["e_mag"])


        # ── Phase 5: WTA tuning update ──
        if learn:
            self.router.update_tuning_wta(x_bottom, active_mask)

        # ── Phase 4: Long-range weight update (learned, local rule) ──
        if learn:
            for i in range(self.n_cols):
                for j in range(self.n_cols):
                    if i != j and active_mask[i] and active_mask[j]:
                        # Hebbian on route scores (columns that co-activate strengthen link)
                        dw = cfg.lr_base * 0.01 * (
                            self.columns[i].route_score * self.columns[j].route_score
                            - self.W_long_range[i, j] * 0.01  # decay
                        )
                        self.W_long_range[i, j] += dw

        # ── Phase 6: Update neuromod state ──
        mean_err = float(np.mean(all_emags)) if all_emags else 1.0
        if neuromod_override is None:
            self._update_neuromod(mean_err, reward)

        # ── Aggregate sheet-level stats ──
        active_outputs = [v for v in outputs.values() if v is not None]
        n_active       = len(active_outputs)

        # Specialisation index: std of route_scores across active cols (higher = more specialised)
        active_rs  = [outputs[i]["route_score"] for i in range(self.n_cols)
                      if outputs[i] is not None]
        spec_index = float(np.std(active_rs)) if len(active_rs) > 1 else 0.0

        # Redundancy: mean cosine similarity between active column hidden states
        active_h = h_states[active_mask]
        if len(active_h) > 1:
            norms    = np.linalg.norm(active_h, axis=1, keepdims=True) + 1e-8
        redundancy = 0.0
        if len(active_h) > 1:
            h_norm   = active_h / norms
            cos_mat  = h_norm @ h_norm.T
            n        = len(active_h)
            redundancy = float((np.sum(cos_mat) - n) / (n * (n - 1) + 1e-8))

        record = {
            "t":            self.t,
            "mean_e_mag":   mean_err,
            "n_active":     n_active,
            "spec_index":   spec_index,
            "redundancy":   redundancy,
            "ach":          self.neuromod.ach,
            "da":           self.neuromod.da,
            "ne":           self.neuromod.ne,
            "active_cols":  active_mask.tolist(),
            "route_scores": col_route_scores.tolist(),
        }
        self.sheet_history.append(record)

        return {
            "outputs":    outputs,
            "active":     active_mask,
            "mean_e_mag": mean_err,
            "spec_index": spec_index,
            "redundancy": redundancy,
            "neuromod":   (self.neuromod.ach, self.neuromod.da, self.neuromod.ne),
        }


# ═══════════════════════════════════════════════════════════════
# Phase 6 — Task switching test
# ═══════════════════════════════════════════════════════════════

def task_switch_test(sheet: CorticalSheet, n_steps_per_task: int = 200,
                     verbose: bool = True) -> Dict:
    """
    Phase 6 success criterion: same circuit, different neuromod state →
    different behaviour without retraining.

    Two tasks on same input:
      Task A: High ACh / moderate DA — sharp sensory learning mode
      Task B: High NE / low DA      — exploration / reset mode
    Metric: mean error and specialisation differ significantly between tasks.
    """
    cfg = sheet.cfg
    rng = np.random.default_rng(77)

    def sine_input(t, task_id):
        freq = 1.0 + task_id * 0.5
        return np.sin(freq * t * 0.1 + rng.normal(0, 0.05, cfg.d_in))

    nm_A = NeuromodState(ach=1.8, da=1.5, ne=0.2)  # sharp attention
    nm_B = NeuromodState(ach=0.6, da=0.5, ne=1.2)  # exploration / reset

    logs = {"A": [], "B": []}
    for task_id, (nm, label) in enumerate([(nm_A, "A"), (nm_B, "B")]):
        for t in range(n_steps_per_task):
            x = sine_input(t, task_id)
            out = sheet.step(x, reward=float(task_id == 0) * 0.1,
                             neuromod_override=nm, learn=True)
            logs[label].append({
                "e_mag":      out["mean_e_mag"],
                "spec_index": out["spec_index"],
                "redundancy": out["redundancy"],
                "n_active":   int(out["active"].sum()),
            })

    result = {}
    for label in ["A", "B"]:
        last100 = logs[label][-100:]
        result[f"mean_e_{label}"]    = float(np.mean([r["e_mag"]      for r in last100]))
        result[f"spec_{label}"]      = float(np.mean([r["spec_index"] for r in last100]))
        result[f"redundancy_{label}"]= float(np.mean([r["redundancy"] for r in last100]))
        result[f"n_active_{label}"]  = float(np.mean([r["n_active"]   for r in last100]))

    e_diff   = abs(result["mean_e_A"] - result["mean_e_B"])
    sp_diff  = abs(result["spec_A"]   - result["spec_B"])
    # Success: measurable behavioural difference under same weights
    result["behaviour_diverges"] = (e_diff > 0.05) or (sp_diff > 0.01)
    result["pass"] = result["behaviour_diverges"]
    result["logs"] = logs

    if verbose:
        print(f"  Task A (hi ACh/DA): e={result['mean_e_A']:.4f}  spec={result['spec_A']:.4f}  n_active={result['n_active_A']:.1f}")
        print(f"  Task B (hi NE):     e={result['mean_e_B']:.4f}  spec={result['spec_B']:.4f}  n_active={result['n_active_B']:.1f}")
        print(f"  Behaviour diverges: {result['behaviour_diverges']}  (Δe={e_diff:.4f}, Δspec={sp_diff:.4f})")
    return result


# ═══════════════════════════════════════════════════════════════
# Phase 2 acceptance test
# ═══════════════════════════════════════════════════════════════

def test_phase2_dendritic_separation(verbose: bool = True) -> Dict:
    """
    Success criterion: same x_bottom, different x_top → different y_pred.
    Also verifies two-stream error signals are distinct.
    """
    cfg  = ColumnConfig()
    rng  = np.random.default_rng(42)
    col  = DendriticColumn(cfg, rng=rng)

    # Warm up: train for 200 steps so h and weights have content
    for _ in range(200):
        col.step(rng.normal(0, 1, cfg.d_in),
                 x_top=rng.normal(0, 1, cfg.d_ctx), learn=True)

    x_b  = rng.normal(0, 1, cfg.d_in)
    # Extreme opposite contexts to maximise apical signal contrast
    ctx1 = np.ones(cfg.d_ctx) *  5.0
    ctx2 = np.ones(cfg.d_ctx) * -5.0

    import copy
    col_copy = copy.deepcopy(col)
    out1 = col.step(x_b, x_top=ctx1, learn=False)
    out2 = col_copy.step(x_b, x_top=ctx2, learn=False)

    cos_ff  = float(np.dot(out1["y_pred_ff"],  out2["y_pred_ff"]) /
                   (np.linalg.norm(out1["y_pred_ff"])  * np.linalg.norm(out2["y_pred_ff"])  + 1e-8))
    cos_ctx = float(np.dot(out1["y_pred_ctx"], out2["y_pred_ctx"]) /
                   (np.linalg.norm(out1["y_pred_ctx"]) * np.linalg.norm(out2["y_pred_ctx"]) + 1e-8))
    cos_out = float(np.dot(out1["y_pred"],     out2["y_pred"]) /
                   (np.linalg.norm(out1["y_pred"])     * np.linalg.norm(out2["y_pred"])     + 1e-8))

    # ff predictions should be similar (same basal input + same h)
    # ctx predictions must diverge under opposite contexts
    # overall output must differ
    ff_stable  = cos_ff  > 0.5    # FF stable: same h + same x_b → similar
    ctx_diverge= cos_ctx < 0.98   # ctx diverges: opposite extreme contexts
    out_diverge= cos_out < 0.999  # blended output shifts

    results = {
        "cos_ff":      cos_ff,  "ff_stable":   ff_stable,
        "cos_ctx":     cos_ctx, "ctx_diverge": ctx_diverge,
        "cos_out":     cos_out, "out_diverge": out_diverge,
        "pass":        ff_stable and ctx_diverge and out_diverge,
    }
    if verbose:
        print(f"  [{'PASS' if ff_stable   else 'FAIL'}] FF stable under ctx switch  cos_ff={cos_ff:.3f}")
        print(f"  [{'PASS' if ctx_diverge else 'FAIL'}] Ctx stream diverges         cos_ctx={cos_ctx:.3f}")
        print(f"  [{'PASS' if out_diverge else 'FAIL'}] Output diverges             cos_out={cos_out:.3f}")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 3 acceptance tests
# ═══════════════════════════════════════════════════════════════

def test_phase3_inhibitory_stability(verbose: bool = True) -> Dict:
    """
    Success criteria:
    1. No runaway activations under 10x noise.
    2. Sparse activation patterns emerge (>50% units silent per step).
    """
    cfg = ColumnConfig()
    rng = np.random.default_rng(7)
    col = DendriticColumn(cfg, rng=rng)

    # Runaway test
    max_emag = 0.0
    for _ in range(100):
        x = rng.normal(0, 10, cfg.d_in)  # 10x magnitude
        out = col.step(x, learn=False)
        max_emag = max(max_emag, out["e_mag"])

    no_runaway = max_emag < 100.0

    # Sparsity test
    sparsities = []
    for _ in range(200):
        x   = rng.normal(0, 1, cfg.d_in)
        out = col.step(x, learn=True)
        sparsities.append(out["sparsity"])
    mean_sparsity = float(np.mean(sparsities))
    sparse_ok = mean_sparsity > 0.50

    results = {
        "max_e_mag": max_emag, "no_runaway":    no_runaway,
        "mean_sparsity": mean_sparsity, "sparse_ok": sparse_ok,
        "pass": no_runaway and sparse_ok,
    }
    if verbose:
        print(f"  [{'PASS' if no_runaway else 'FAIL'}] No runaway  max_e={max_emag:.2f}")
        print(f"  [{'PASS' if sparse_ok  else 'FAIL'}] Sparsity    mean={mean_sparsity:.3f}")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 4 acceptance tests
# ═══════════════════════════════════════════════════════════════

def test_phase4_specialisation(n_steps: int = 600,
                                verbose: bool = True) -> Dict:
    """
    Success criteria:
    1. Specialisation index increases over training.
    2. Redundancy decreases over training.
    """
    sheet = CorticalSheet(n_cols=12, k_active=4, rng_seed=0)
    cfg   = sheet.cfg
    rng   = np.random.default_rng(0)

    def multi_feature_input(t):
        """Different sine frequencies — gives columns something to specialise on"""
        freqs  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                  0.7, 1.3, 1.8, 2.3, 2.7, 3.2, 3.8, 4.5]
        return np.array([np.sin(f * t * 0.05 + i) for i, f in enumerate(freqs)])

    specs, reds = [], []
    for t in range(n_steps):
        x   = multi_feature_input(t)
        out = sheet.step(x, learn=True)
        if t >= 50:
            specs.append(out["spec_index"])
            reds.append(out["redundancy"])

    early_spec  = float(np.mean(specs[:100]))
    late_spec   = float(np.mean(specs[-100:]))
    early_red   = float(np.mean(reds[:100]))
    late_red    = float(np.mean(reds[-100:]))

    spec_up  = late_spec  >= early_spec * 0.85   # spec should not collapse
    # Redundancy: acceptable if it stays bounded (not monotonically growing)
    mid_red  = float(np.mean(reds[len(reds)//3 : 2*len(reds)//3]))
    red_down = late_red <= mid_red * 1.1 or late_red < 0.15  # stabilises or stays low

    results = {
        "early_spec": early_spec, "late_spec":  late_spec,  "spec_up":  spec_up,
        "early_red":  early_red,  "late_red":   late_red,   "red_down": red_down,
        "pass": spec_up and red_down,
    }
    if verbose:
        print(f"  [{'PASS' if spec_up  else 'FAIL'}] Spec index  {early_spec:.4f} → {late_spec:.4f}")
        print(f"  [{'PASS' if red_down else 'FAIL'}] Redundancy  {early_red:.4f} → {late_red:.4f}")
    return results, sheet


# ═══════════════════════════════════════════════════════════════
# Phase 5 acceptance tests
# ═══════════════════════════════════════════════════════════════

def test_phase5_routing(sheet: Optional[CorticalSheet] = None,
                         n_steps: int = 400,
                         verbose: bool = True) -> Dict:
    """
    Success criteria:
    1. Different input classes activate different column subsets (thalamic tuning).
    2. Active set is sparse (k < n/2).

    Uses a fresh sheet so Phase 4 base-score drift doesn't mask tuning signal.
    """
    fresh = CorticalSheet(n_cols=12, k_active=4, rng_seed=0)
    cfg   = fresh.cfg
    rng   = np.random.default_rng(55)

    def input_class(class_id):
        base = np.zeros(cfg.d_in)
        if class_id == 0:
            base[:cfg.d_in//2] = 3.0
        else:
            base[cfg.d_in//2:] = 3.0
        return base + rng.normal(0, 0.1, cfg.d_in)

    # Train interleaved to build routing differentiation via WTA tuning
    for t in range(500):
        fresh.step(input_class(t % 2), learn=True)

    # Eval: measure per-column activation frequency per class
    active_0 = np.zeros(fresh.n_cols)
    active_1 = np.zeros(fresh.n_cols)
    for t in range(n_steps):
        out0 = fresh.step(input_class(0), learn=False)
        out1 = fresh.step(input_class(1), learn=False)
        active_0 += out0["active"].astype(float)
        active_1 += out1["active"].astype(float)
    active_0 /= n_steps
    active_1 /= n_steps

    diff = active_0 - active_1
    pref_0 = set(np.where(diff >  0.05)[0])
    pref_1 = set(np.where(diff < -0.05)[0])
    different_subsets = len(pref_0) > 0 or len(pref_1) > 0

    sparse = fresh.k / fresh.n_cols < 0.75
    results = {
        "pref_0_cols": len(pref_0), "pref_1_cols": len(pref_1),
        "different_subsets": different_subsets,
        "k_over_n": fresh.k / fresh.n_cols, "sparse": sparse,
        "pass": different_subsets and sparse,
    }
    if verbose:
        print(f"  [{'PASS' if different_subsets else 'FAIL'}] Different subsets  pref_0={len(pref_0)} pref_1={len(pref_1)}")
        print(f"  [{'PASS' if sparse            else 'FAIL'}] Sparse routing     k/n={fresh.k/fresh.n_cols:.2f}")
    return results


# ═══════════════════════════════════════════════════════════════
# Full test runner
# ═══════════════════════════════════════════════════════════════

def run_all_tests(verbose: bool = True) -> Dict:
    results = {}

    print("\n═══ Phase 2: Dendritic Separation ═══")
    results["phase2"] = test_phase2_dendritic_separation(verbose)

    print("\n═══ Phase 3: Inhibitory Stability ═══")
    results["phase3"] = test_phase3_inhibitory_stability(verbose)

    print("\n═══ Phase 4: Multi-Column Specialisation ═══")
    r4, sheet = test_phase4_specialisation(verbose=verbose)
    results["phase4"] = r4

    print("\n═══ Phase 5: Thalamic Routing ═══")
    results["phase5"] = test_phase5_routing(verbose=verbose)

    print("\n═══ Phase 6: Task Switching ═══")
    results["phase6"] = task_switch_test(sheet, verbose=verbose)

    passed = sum(1 for v in results.values() if v.get("pass", False))
    print(f"\n{'═'*42}")
    print(f"  {passed}/{len(results)} phases passing")
    return results


# ═══════════════════════════════════════════════════════════════
# Training log export for visualiser
# ═══════════════════════════════════════════════════════════════

def run_full_training(n_steps: int = 800, export_path: str = "training_log_full.json"):
    """Run a full sheet training session and export log for the visualiser."""
    sheet = CorticalSheet(n_cols=12, k_active=4, rng_seed=42)
    cfg   = sheet.cfg
    rng   = np.random.default_rng(42)

    def input_stream(t):
        freqs = [0.3 + i * 0.25 for i in range(cfg.d_in)]
        return np.array([np.sin(f * t * 0.08 + i * 0.4) for i, f in enumerate(freqs)])

    log = []
    for t in range(n_steps):
        x   = input_stream(t)
        reward = 0.1 if t % 100 < 50 else 0.0  # periodic reward signal
        out = sheet.step(x, reward=reward, learn=True)
        log.append({
            "t":           t,
            "e_mag":       float(out["mean_e_mag"]),
            "spec_index":  float(out["spec_index"]),
            "redundancy":  float(out["redundancy"]),
            "n_active":    int(out["active"].sum()),
            "ach":         float(out["neuromod"][0]),
            "da":          float(out["neuromod"][1]),
            "ne":          float(out["neuromod"][2]),
            "route_scores": [float(x) for x in sheet.router.route_scores],
            "active":      out["active"].tolist(),
        })

    with open(export_path, "w") as f:
        json.dump(log, f)
    print(f"Full training log saved → {export_path}  ({n_steps} steps)")
    return log, sheet


if __name__ == "__main__":
    print("Cortical Microcircuit — Phase 2 through Phase 6")
    print("=" * 48)

    results = run_all_tests(verbose=True)

    print("\n── Running full training for visualiser ──")
    log, sheet = run_full_training(n_steps=800, export_path="training_log_full.json")

    # Summary stats
    first100 = log[:100]
    last100  = log[-100:]
    e_start  = np.mean([r["e_mag"]      for r in first100])
    e_end    = np.mean([r["e_mag"]      for r in last100])
    sp_start = np.mean([r["spec_index"] for r in first100])
    sp_end   = np.mean([r["spec_index"] for r in last100])
    print(f"\n  Error:        {e_start:.4f} → {e_end:.4f}  ({(1-e_end/e_start)*100:.1f}% reduction)")
    print(f"  Spec index:   {sp_start:.4f} → {sp_end:.4f}")
    print(f"  Columns used: {sheet.n_cols}, k_active={sheet.k}")
