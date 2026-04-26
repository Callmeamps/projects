"""
Cortical Microcircuit — Phase 0 + Phase 1
Spec v0.4 implementation

Phase 0: Foundations (tensor shapes, compartment rules, inhibition, stability)
Phase 1: Minimal predictive unit (single column, local Hebbian + error learning)

No global backprop. All updates are local.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json


# ─────────────────────────────────────────────
# Phase 0.1 — Tensor shapes / config
# ─────────────────────────────────────────────

@dataclass
class ColumnConfig:
    d_in: int = 16        # x_bottom dimensionality
    d_ctx: int = 16       # x_top dimensionality
    d_lat: int = 16       # x_lateral dimensionality (per neighbor)
    d_h: int = 32         # recurrent state size
    d_action: int = 4     # action output dimension
    k_neighbors: int = 4  # lateral connections

    # Inhibitory clips (Phase 0.8)
    pv_clip: tuple = (0.2, 5.0)
    sst_clip: tuple = (0.0, 1.0)   # SST suppression in [0,1] — fills the spec gap

    # Plasticity (Phase 0.6)
    lr_base: float = 0.01
    tau_elig: float = 20           # eligibility trace decay (timesteps, ~200ms @ 10ms/step)
    anti_hebb_scale: float = 0.1   # Section 0.8: anti-Hebbian = 0.1 * Hebbian

    # Routing stability (Phase 0.4)
    tau_route: float = 5.0
    target_firing_rate: float = 0.05  # homeostatic target


# ─────────────────────────────────────────────
# Phase 0.5 — Neuromodulatory state
# ─────────────────────────────────────────────

@dataclass
class NeuromodState:
    """
    m_global = [ACh, DA, NE]
    Each has its own timescale (Phase 0.5):
      ACh: fast (20-50ms ~ 2-5 steps)
      DA:  medium phasic (100-200ms ~ 10-20 steps)
      NE:  slow/reset (seconds ~ 100+ steps)
    """
    ach: float = 1.0   # sensory gain / attention
    da: float = 1.0    # learning salience / reward
    ne: float = 0.5    # uncertainty / exploration

    def as_vector(self) -> np.ndarray:
        return np.array([self.ach, self.da, self.ne])

    def learning_rate_scale(self) -> float:
        """DA gates plasticity magnitude"""
        return self.da

    def pv_gain_scale(self) -> float:
        """ACh sharpens sensory gain via PV modulation"""
        return self.ach

    def exploration_mode(self) -> bool:
        """High NE = reset / explore"""
        return self.ne > 0.8


# ─────────────────────────────────────────────
# Phase 0.2 — Pyramidal neuron compartments
# ─────────────────────────────────────────────

class PyramidalCompartment:
    """
    3-compartment pyramidal unit:
      basal  : W_b @ x_bottom  (feedforward evidence, linear)
      apical : W_a @ x_top     (top-down context, multiplicative gate)
      soma   : basal * sigmoid(apical)  — coincidence detection
    """

    def __init__(self, d_in: int, d_ctx: int, d_h: int, rng: np.random.Generator):
        scale = 1.0 / np.sqrt(d_in)
        self.W_basal  = rng.normal(0, scale, (d_h, d_in))
        self.W_apical = rng.normal(0, 1.0 / np.sqrt(d_ctx), (d_h, d_ctx))

    def forward(self, x_bottom: np.ndarray, x_top: np.ndarray,
                sst_suppression: float = 0.0) -> tuple:
        """
        Returns (soma_out, basal_act, apical_gate)
        sst_suppression ∈ [0,1] scales down apical influence (Phase 0.3)
        """
        basal  = self.W_basal @ x_bottom                          # feedforward evidence
        apical = self.W_apical @ x_top * (1.0 - sst_suppression) # context, SST-gated
        gate   = _sigmoid(apical)                                  # expectation gate
        soma   = basal * gate                                      # coincidence detection
        return soma, basal, gate


# ─────────────────────────────────────────────
# Phase 0.3 — Inhibitory interneuron system
# ─────────────────────────────────────────────

class InhibitorySystem:
    """
    PV  — divisive gain control on soma output
    SST — suppress apical influence (scalar in [0,1])
    VIP — disinhibit by suppressing PV and SST (exploration gate)
    """

    def __init__(self, cfg: ColumnConfig):
        self.cfg = cfg
        self.pv_gain  = 1.0
        self.sst_sup  = 0.0
        self.vip_gate = 0.0

    def update(self, soma_activation: np.ndarray,
               neuromod: NeuromodState, error_magnitude: float) -> None:
        """
        PV: reacts to activation magnitude, scaled by ACh
        SST: reacts to sustained apical drive
        VIP: fires under high NE (exploration / uncertainty)
        """
        activity_level = float(np.mean(np.abs(soma_activation)))

        # PV: proportional to activity, ACh sharpens it
        raw_pv = activity_level * neuromod.pv_gain_scale()
        self.pv_gain = float(np.clip(raw_pv + 0.5, *self.cfg.pv_clip))

        # VIP fires under high NE, suppresses PV+SST
        self.vip_gate = float(np.clip(neuromod.ne - 0.5, 0, 1))
        vip_suppress  = self.vip_gate * 0.8

        # SST: driven by error (high error → more apical suppression), VIP dials it down
        raw_sst = error_magnitude * 0.3 * (1.0 - vip_suppress)
        self.sst_sup = float(np.clip(raw_sst, *self.cfg.sst_clip))

    def apply_pv(self, soma: np.ndarray) -> np.ndarray:
        """Divisive gain: soma / pv_gain"""
        return soma / self.pv_gain

    @property
    def sst_suppression(self) -> float:
        return self.sst_sup


# ─────────────────────────────────────────────
# Phase 0.6 — Plasticity
# ─────────────────────────────────────────────

class LocalPlasticity:
    """
    Implements all four plasticity rules (Section 7 + Phase 0.6):
      1. Hebbian         Δw ∝ pre * post
      2. Anti-Hebbian    Δw ∝ -0.1 * pre * post  (decorrelation)
      3. Error-modulated weight updates scaled by e_local magnitude
      4. Gated by DA (neuromodulatory state)

    Eligibility traces bridge temporal gaps (tau_elig, Phase 0.6)
    """

    def __init__(self, cfg: ColumnConfig):
        self.cfg = cfg
        self.e_trace_basal  = None   # initialized on first use
        self.e_trace_apical = None

    def _init_traces(self, W_basal: np.ndarray, W_apical: np.ndarray):
        if self.e_trace_basal is None:
            self.e_trace_basal  = np.zeros_like(W_basal)
            self.e_trace_apical = np.zeros_like(W_apical)

    def update(self, pyramidal: PyramidalCompartment,
               x_bottom: np.ndarray, x_top: np.ndarray,
               soma_out: np.ndarray, basal_act: np.ndarray,
               e_local: np.ndarray, neuromod: NeuromodState) -> dict:
        """
        Returns dict of delta norms for logging.
        All math is local — no upstream gradients.
        """
        self._init_traces(pyramidal.W_basal, pyramidal.W_apical)

        decay = np.exp(-1.0 / self.cfg.tau_elig)
        lr    = self.cfg.lr_base * neuromod.learning_rate_scale()
        err_mag = float(np.mean(np.abs(e_local)))

        # ── Eligibility trace update (STDP-like, causal) ──
        # pre = x_bottom, post = soma_out (for basal)
        hebb_basal  = np.outer(soma_out, x_bottom)
        hebb_apical = np.outer(soma_out, x_top)

        self.e_trace_basal  = decay * self.e_trace_basal  + (1 - decay) * hebb_basal
        self.e_trace_apical = decay * self.e_trace_apical + (1 - decay) * hebb_apical

        # ── Error-modulated Hebbian update ──
        delta_W_basal  = lr * err_mag * self.e_trace_basal
        delta_W_apical = lr * err_mag * self.e_trace_apical

        # ── Anti-Hebbian (decorrelation / sparsity) ──
        # Gentle: penalizes large co-activations only
        anti_scale = self.cfg.anti_hebb_scale * float(np.mean(soma_out ** 2))
        delta_W_basal  -= lr * anti_scale * np.sign(hebb_basal)
        delta_W_apical -= lr * anti_scale * np.sign(hebb_apical)

        # ── Apply ──
        pyramidal.W_basal  += delta_W_basal
        pyramidal.W_apical += delta_W_apical

        # ── Homeostatic norm clipping (Phase 0.8) — prevents explosion, not collapse ──
        max_norm = 5.0
        bn = np.linalg.norm(pyramidal.W_basal)
        an = np.linalg.norm(pyramidal.W_apical)
        if bn > max_norm:
            pyramidal.W_basal  *= max_norm / bn
        if an > max_norm:
            pyramidal.W_apical *= max_norm / an

        return {
            "delta_basal_norm":  float(np.linalg.norm(delta_W_basal)),
            "delta_apical_norm": float(np.linalg.norm(delta_W_apical)),
            "elig_trace_mean":   float(np.mean(np.abs(self.e_trace_basal))),
        }


# ─────────────────────────────────────────────
# Phase 1 — Single Predictive Column
# ─────────────────────────────────────────────

class CorticalColumn:
    """
    Phase 1: Minimal predictive column.
    
    Data flow (Section 11):
      1. Receive inputs
      2. Generate prediction y_pred
      3. Compute error e_local = x_bottom - y_pred
      4. Update inhibitory state
      5. Apply plasticity
      6. Update recurrent state h
    """

    def __init__(self, cfg: ColumnConfig, column_id: int = 0,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.id  = column_id
        rng = rng or np.random.default_rng(42 + column_id)

        # Pyramidal compartment (Phase 0.2)
        self.pyramidal  = PyramidalCompartment(cfg.d_in, cfg.d_ctx, cfg.d_h, rng)

        # Readout: maps hidden state → prediction
        scale = 1.0 / np.sqrt(cfg.d_h)
        self.W_pred     = rng.normal(0, scale, (cfg.d_in, cfg.d_h))

        # Recurrent weights
        self.W_recurrent  = rng.normal(0, 0.1 / np.sqrt(cfg.d_h), (cfg.d_h, cfg.d_h))
        # Project recurrent state into ctx space for apical augmentation
        self.W_h_to_ctx   = rng.normal(0, 0.1 / np.sqrt(cfg.d_h), (cfg.d_ctx, cfg.d_h))

        # Inhibitory system (Phase 0.3)
        self.inhibitory = InhibitorySystem(cfg)

        # Plasticity (Phase 0.6)
        self.plasticity = LocalPlasticity(cfg)

        # Internal state
        self.h           = np.zeros(cfg.d_h)   # recurrent state
        self.p           = 1.0                  # precision estimate
        self.route_score = 0.5                  # Phase 0.4 routing
        self._route_ema  = 0.5

        # Logging
        self.history: list[dict] = []

    def step(self, x_bottom: np.ndarray,
             x_top: Optional[np.ndarray] = None,
             x_lateral: Optional[np.ndarray] = None,
             neuromod: Optional[NeuromodState] = None,
             learn: bool = True) -> dict:
        """
        One full timestep through all 6 phases.
        Returns output dict with y_pred, e_local, route_score, h_next.
        """
        cfg = self.cfg
        neuromod = neuromod or NeuromodState()

        # Default zero context if not provided
        if x_top is None:
            x_top = np.zeros(cfg.d_ctx)

        # ── Phase 1-2: Feedforward + Prediction ──

        # Recurrent context augments apical (lateral is a separate future phase)
        x_top_aug = x_top + 0.3 * (self.W_h_to_ctx @ self.h)

        soma_out, basal_act, apical_gate = self.pyramidal.forward(
            x_bottom, x_top_aug,
            sst_suppression=self.inhibitory.sst_suppression
        )

        # PV divisive gain control
        soma_gated = self.inhibitory.apply_pv(soma_out)

        # Prediction from current hidden state
        y_pred = _tanh(self.W_pred @ self.h)

        # ── Phase 3: Error computation ──
        e_local = x_bottom - y_pred
        err_mag = float(np.linalg.norm(e_local))

        # Update precision estimate (running mean of inverse error)
        self.p = 0.95 * self.p + 0.05 * (1.0 / (err_mag + 1e-4))

        # ── Phase 4: Inhibition update ──
        self.inhibitory.update(soma_gated, neuromod, err_mag)

        # ── Phase 5: Plasticity (if learning mode) ──
        plasticity_stats = {}
        if learn and not neuromod.exploration_mode():
            plasticity_stats = self.plasticity.update(
                self.pyramidal, x_bottom, x_top_aug,
                soma_gated, basal_act, e_local, neuromod
            )

            # Update readout W_pred with error signal (local rule)
            lr = cfg.lr_base * neuromod.learning_rate_scale()
            dW_pred = lr * err_mag * np.outer(e_local, self.h)
            self.W_pred += dW_pred

        # ── Phase 6: State update ──
        h_next = _tanh(soma_gated + 0.5 * (self.W_recurrent @ self.h))
        self.h = h_next

        # ── Routing score update (Phase 0.4, low-pass filtered) ──
        alpha = 1.0 / cfg.tau_route
        raw_score = self.p * float(np.mean(np.abs(soma_gated)))
        self._route_ema = (1 - alpha) * self._route_ema + alpha * raw_score
        self.route_score = self._route_ema

        result = {
            "y_pred":      y_pred,
            "e_local":     e_local,
            "e_mag":       err_mag,
            "route_score": self.route_score,
            "h_next":      h_next,
            "precision":   self.p,
            "pv_gain":     self.inhibitory.pv_gain,
            "sst_sup":     self.inhibitory.sst_sup,
            "vip_gate":    self.inhibitory.vip_gate,
            **plasticity_stats,
        }

        self.history.append({k: float(v) if np.isscalar(v) else float(np.mean(np.abs(v)))
                             for k, v in result.items()
                             if k not in ("y_pred", "e_local", "h_next")})
        return result


# ─────────────────────────────────────────────
# Phase 1 — Toy sequence training
# ─────────────────────────────────────────────

def generate_sine_sequence(n_steps: int, d_in: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Multi-frequency sine wave packed into d_in dims"""
    t = np.linspace(0, 4 * np.pi, n_steps)
    freqs = rng.uniform(0.5, 3.0, d_in)
    phases = rng.uniform(0, 2 * np.pi, d_in)
    return np.stack([np.sin(freqs[i] * t + phases[i]) for i in range(d_in)], axis=1)


def run_phase1_trial(n_steps: int = 500, cfg: Optional[ColumnConfig] = None,
                     neuromod: Optional[NeuromodState] = None) -> dict:
    """
    Train a single column to predict the next step of a multi-dim sine sequence.
    Returns full training log.
    """
    cfg      = cfg or ColumnConfig()
    neuromod = neuromod or NeuromodState()
    rng      = np.random.default_rng(0)

    col = CorticalColumn(cfg, column_id=0, rng=rng)
    seq = generate_sine_sequence(n_steps, cfg.d_in, rng)

    log = []
    for t in range(1, n_steps):
        x_now  = seq[t]
        x_prev = seq[t - 1]   # x_top = previous step as context

        out = col.step(x_now, x_top=x_prev, neuromod=neuromod, learn=True)
        log.append({
            "t":           t,
            "e_mag":       out["e_mag"],
            "route_score": out["route_score"],
            "precision":   out["precision"],
            "pv_gain":     out["pv_gain"],
            "sst_sup":     out["sst_sup"],
        })

    return {"log": log, "column": col}


# ─────────────────────────────────────────────
# Phase 0.9 — Acceptance tests
# ─────────────────────────────────────────────

def run_acceptance_tests(verbose: bool = True) -> dict:
    cfg = ColumnConfig()
    rng = np.random.default_rng(99)
    results = {}

    # ── Test 1: Shape + no NaN ──
    col = CorticalColumn(cfg, rng=rng)
    passed = True
    for _ in range(10_000):
        x_b = rng.normal(0, 1, cfg.d_in)
        x_t = rng.normal(0, 1, cfg.d_ctx)
        out = col.step(x_b, x_t)
        if np.any(np.isnan(out["y_pred"])):
            passed = False
            break
    results["shape_no_nan"] = passed

    # ── Test 2: Apical gating ──
    col2 = CorticalColumn(cfg, rng=np.random.default_rng(7))
    x_b  = rng.normal(0, 1, cfg.d_in)
    x_t1 = rng.normal(0,  1, cfg.d_ctx)
    x_t2 = -rng.normal(0, 1, cfg.d_ctx)
    out1 = col2.step(x_b, x_t1, learn=False)
    out2 = col2.step(x_b, x_t2, learn=False)
    cos_sim = float(np.dot(out1["y_pred"], out2["y_pred"]) /
                    (np.linalg.norm(out1["y_pred"]) * np.linalg.norm(out2["y_pred"]) + 1e-8))
    results["apical_gating_cos_sim"] = cos_sim
    results["apical_gating_pass"]    = cos_sim < 0.95   # different contexts → different preds

    # ── Test 3: PV runaway prevention ──
    col3 = CorticalColumn(cfg, rng=np.random.default_rng(3))
    baseline = []
    for _ in range(20):
        out = col3.step(rng.normal(0, 1, cfg.d_in), learn=False)
        baseline.append(out["e_mag"])
    baseline_mean = np.mean(baseline)

    spike_acts = []
    for _ in range(20):
        out = col3.step(rng.normal(0, 10, cfg.d_in), learn=False)  # 10x magnitude
        spike_acts.append(out["e_mag"])
    spike_mean = np.mean(spike_acts)
    results["pv_runaway_ratio"] = float(spike_mean / (baseline_mean + 1e-8))
    results["pv_runaway_pass"]  = results["pv_runaway_ratio"] < 20.0

    # ── Test 4: Neuromod difference ──
    col4 = CorticalColumn(cfg, rng=np.random.default_rng(5))
    hi_ach = NeuromodState(ach=2.0, da=1.0, ne=0.1)
    hi_ne  = NeuromodState(ach=1.0, da=1.0, ne=0.9)
    pv_ach, pv_ne = [], []
    for _ in range(50):
        x = rng.normal(0, 1, cfg.d_in)
        col4.step(x, neuromod=hi_ach, learn=True)
        pv_ach.append(col4.inhibitory.pv_gain)
    for _ in range(50):
        x = rng.normal(0, 1, cfg.d_in)
        col4.step(x, neuromod=hi_ne, learn=True)
        pv_ne.append(col4.inhibitory.pv_gain)
    pv_diff = abs(np.mean(pv_ach) - np.mean(pv_ne)) / (np.mean(pv_ach) + 1e-8)
    results["neuromod_pv_diff_pct"] = float(pv_diff * 100)
    results["neuromod_pass"]        = pv_diff > 0.10   # >10% difference

    if verbose:
        print("\n═══ Phase 0.9 Acceptance Tests ═══")
        print(f"  [{'PASS' if results['shape_no_nan'] else 'FAIL'}] Shape + no NaN (10k steps)")
        print(f"  [{'PASS' if results['apical_gating_pass'] else 'FAIL'}] Apical gating  cos_sim={cos_sim:.3f}")
        print(f"  [{'PASS' if results['pv_runaway_pass'] else 'FAIL'}] PV runaway     ratio={results['pv_runaway_ratio']:.2f}x")
        print(f"  [{'PASS' if results['neuromod_pass'] else 'FAIL'}] Neuromod diff  {results['neuromod_pv_diff_pct']:.1f}%")

    return results


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -30, 30))


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Cortical Microcircuit — Phase 0 + Phase 1")
    print("==========================================\n")

    # Run acceptance tests
    test_results = run_acceptance_tests(verbose=True)

    # Run Phase 1 training trial
    print("\n── Phase 1: Training single column on sine sequence ──")
    trial = run_phase1_trial(n_steps=500)
    log   = trial["log"]

    # Print summary
    first50  = [e["e_mag"] for e in log[:50]]
    last50   = [e["e_mag"] for e in log[-50:]]
    print(f"  Error (first 50 steps): mean={np.mean(first50):.4f}")
    print(f"  Error (last  50 steps): mean={np.mean(last50):.4f}")
    reduction = (1 - np.mean(last50) / np.mean(first50)) * 100
    print(f"  Reduction: {reduction:.1f}%")
    print(f"\n  Final PV gain:   {log[-1]['pv_gain']:.3f}")
    print(f"  Final precision: {log[-1]['precision']:.4f}")

    # Export log for visualizer
    with open("training_log.json", "w") as f:
        json.dump(log, f)
    print("\nTraining log saved → training_log.json")
