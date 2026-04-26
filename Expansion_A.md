# Unified Predictive Cortex & Swarm Agent Specification (`spec.md`)

## 1. Purpose & Vision
Define a synthetic cortical microcircuit architecture that operates as a **continuous-learning, sparse-routing, predictive-control swarm**. The system predicts, reasons, acts, and adapts online without global backpropagation, maintaining multi-timescale memory and supporting diverse environments. The target is **not a giant monolithic model**, but a coordinated system of repeated predictive units that scale via swarm topology, sparse communication, and functional specialization.

## 2. Core Design Principles
- **Functional equivalence over anatomical replication**: Biology is a reference, not a blueprint.
- **Local intelligence**: Prediction, correction, routing, and plasticity occur locally.
- **Closed-loop agency**: Prediction shapes action; action consequences shape prediction.
- **Sparse by design**: Default to low activation, low connectivity, and top-k routing.
- **Conscious vs. Subconscious split**: Subconscious columns handle parallel prediction; the conscious workspace handles serial coordination, reasoning, and action selection.
- **No global backprop for inference/learning**: Credit assignment uses local error signals, eligibility traces, and modulatory scaling.

## 3. System Architecture Overview
| Subsystem | Role |
|-----------|------|
| **Predictive Columns** | Core unit for local prediction, error correction, state update, and action proposal |
| **Inhibitory Control Layer** | PV, SST, VIP-style gating for stability, sparsity, and mode switching |
| **Thalamic Routing / Competition** | Selects participating columns via sparse scoring and top-k gating |
| **Global Workspace** | Small shared bottleneck for active context, candidate unification, and cross-column broadcast |
| **Engram Memory Layer** | Fast associative memory for patterns, task fragments, and compressed state-action traces |
| **Adapter / Plasticity Layer** | Tiny fast-adapting parameters (LoRA-like) for task-specific rapid tuning |
| **Planner / Search Loop** | Generates, simulates, scores, and selects candidate actions or thought steps |
| **Environment Interface** | Standardized bridge to chat, game, tool, and sensory/sequence domains |

## 4. Computational Core: Columns & Compartmental Neurons
### 4.1 Neuron Abstraction (Pyramidal Model)
Each computational unit models a pyramidal neuron with three compartments:
- **Basal**: Encodes feedforward/sensory evidence. Drives local prediction.
- **Apical**: Encodes top-down expectation/context. Modulates confidence, gain, and routing priority.
- **Soma**: Integrates basal + apical streams. Produces active representation and low-bandwidth action proposal.

### 4.2 Column Structure
Columns are hierarchical stacks mapping to cortical layers:
- `L4-like`: Primary input encoding
- `L2/3-like`: Integration and lateral mixing
- `L5-like`: Action and output projection
- `L6-like`: Feedback and routing influence

### 4.3 Inputs, State & Outputs
| Category | Signals |
|----------|---------|
| **Inputs** | `x_bottom` (sensory), `x_top` (context), `x_lateral` (neighbors), `r_route` (gate), `m_global` (modulation), `w_task` (mode), `mem_ctx` (retrieved engrams) |
| **Internal State** | `h` (recurrent), `p` (precision/confidence), `g` (inhibitory gain), `e_trace` (eligibility), `u` (action proposal), `k` (workspace key) |
| **Outputs** | `y_pred` (prediction), `e_local` (prediction error), `a_prop` (action/symbol), `s_route` (routing score), `h_next` (updated state), `k_mem`/`v_mem` (memory key/value) |

## 5. Inference Dynamics & Attractor Relaxation
Inference operates as an iterative local mismatch minimization process, analogous to Hopfield energy relaxation.

### 5.1 Local Mismatch & Update
Let `b_i` = basal drive, `a_i` = apical drive, `s_i` = soma state at neuron `i`:
```
ε_i = b_i - a_i
s_i^{t+1} = s_i^t + η(α·b_i + β·a_i + γ·ε_i)
```
- `η`: step size / inference rate
- `α, β, γ`: compartment weighting coefficients
- Bounded by damping, state clipping, or normalization to prevent divergence.

### 5.2 Convergence & Energy
- **Convergence criterion**: `‖s^{t+1} - s^t‖ < δ` or max iteration cap.
- **Energy function**: `E = ½ Σ|b_i - a_i|²`. Inference monotonically reduces `E` under stable parameters.
- Supports pattern completion, noise robustness, and hierarchical prior integration.

## 6. Inhibitory Control & Stability
Inhibition is first-class and dynamically modulated:
| Interneuron Type | Function |
|------------------|----------|
| **PV** | Fast divisive gain control; enforces top-k sparsity; prevents runaway excitation |
| **SST** | Suppresses noisy/unhelpful apical influence; prevents top-down hallucination |
| **VIP** | Disinhibits columns; enables exploration, task switching, and novelty processing |

**Stability Targets**: Sparse activity by default, bounded gain, no runaway firing, graceful explore/exploit switching.

## 7. Routing & Global Workspace
### 7.1 Routing Mechanism
- Distributed scoring → workspace aggregation → hard top-k selection (primary) + soft weighting (secondary).
- **Score signals**: prediction error, novelty, task relevance, reward/usefulness, confidence, memory match strength.
- Acts as the system's functional attention mechanism.

### 7.2 Global Workspace
- Small shared bottleneck collecting top-k column outputs.
- Holds task context, unifies action candidates, consults memory retrieval, stabilizes task mode.
- Broadcasts compact state back to active columns.
- Serves as the bridge between local computation and global behavior.

## 8. Multi-Timescale Memory System
### 8.1 Engram Architecture
- Stores sparse activity keys, compressed values, task fragments, useful action traces, and repeated column coalitions.
- **Retrieval**: Compute sparse key → match against engrams → feed nearest traces to workspace/columns.
- **Lifecycle**: Reinforce when useful → decay when unused → consolidate when repeatedly confirmed.

### 8.2 Memory Timescales & Consolidation
| Type | Analogue | Properties |
|------|----------|------------|
| **Working Memory** | Prefrontal buffers | Current step, short traces, fast decay |
| **Episodic Memory** | Hippocampal analogue | Task events, high plasticity, temporary |
| **Long-Term Memory** | Cortical analogue | Frequently reinforced patterns, stable, compressible |

- **Replay Bridge**: Offline sampling of past traces → re-run through model → reinforce stable patterns. Enables consolidation without global backprop.

## 9. Adaptation & Fast Plasticity Layer
- **Purpose**: Rapid task adaptation without destabilizing the stable core.
- **Mechanism**: Low-rank adapters, fast weights, context-gated plastic modules, tiny side paths.
- **Behavior**: Quick changes, decay/reset when stale, preserve base weights, specialize per environment/task (chat, game, planning, tool use).

## 10. Action Selection & Closed-Loop Control
### 10.1 Action Outputs
Each timestep may emit: tool call, chat token/chunk, movement/game action, memory query, or no-op (if confidence low).

### 10.2 Action Scoring
Candidates scored by: predicted error reduction, reward expectation, novelty usefulness, task alignment, stability risk.

### 10.3 Basal Ganglia Analogue
- Separates **generate** (columns/workspace) from **select** (dedicated selection mechanism).
- Implements go/no-go pathways, reward/habit bias, and commit/suppress behavior.

### 10.4 Closed-Loop Rule
`Action → Environment → New Observation → Prediction/Memory Update`. Outcome error (expected vs actual result) is integrated into local prediction mismatch.

## 11. Neuromodulation & Reward Signaling
### 11.1 Reward Sources
Task success, user approval, environment progress, uncertainty reduction, memory usefulness.

### 11.2 Modulatory States
| Modulator | Function |
|-----------|----------|
| **ACh** | Sharpen sensory gain & attention; increases bottom-up precision |
| **DA** | Reinforce useful action & memory outcomes; boosts reward confidence |
| **NE** | Trigger reset, exploration, or mode switch; signals volatility |

### 11.3 Behavioral Modes
Same network shifts behavior based on modulation: passive chat, active reasoning, exploration, planning, tool use, game control.

## 12. Internal Reasoning & Search
- **Purpose**: Avoid first-plausible-answer bias; compare candidate continuations.
- **Steps**: Propose candidates → simulate briefly with predictive core → score error/reward/stability → select best.
- **Implementation**: Short rollouts, beam-like evaluation, branch-and-score (small depth), task-specific self-check passes.
- Provides compact reasoning without requiring giant parameter scale.

## 13. Environment Interface Layer
Standardized interface per environment:
- **Provides**: observation, action space, reward/feedback, episode reset, optional partial observability/text streams.
- **Categories**:
  - `Chat`: Dialogue, instruction following, tool use, memory writes, clarifying questions.
  - `Game`: Control, planning, reactive decision-making, embodied behavior.
  - `Simulated Agent`: Multi-step autonomy, task execution, tool coordination.
  - `Sensory/Sequence`: Toy training, stability tests, symbolic/patch/time-series streams.

## 14. Swarm Organization & Emergent Specialization
- **Design**: Multiple sheets/tiles/clusters communicate sparsely via workspace and memory retrieval. Bandwidth-limited by design.
- **Specialization**: Emerges through use (language, memory, planning, perception, control, tool use, long-horizon tracking). No hard-coded roles.
- **Scaling**: Achieved via swarm coordination, routing efficiency, and memory reuse, not parameter bloat.

## 15. Learning Rules & Credit Assignment
### 15.1 Required Mix
- Fast local Hebbian updates
- Error-modulated scaling
- Eligibility traces (temporal bridge)
- Local difference signals between layers

### 15.2 Learning Modes
- **Inference-only**: Fixed weights, state relaxation only.
- **Adaptive**: Weights update via local mismatch `ΔW ∝ ε ⊗ μ`.
- **Hybrid**: Inference first, weight update afterward.
- **Optional**: Proxy gradients via prediction error routing, homeostatic normalization.

### 15.3 Goal
Approximate deep coordination and long-horizon credit assignment **without global backpropagation**.

## 16. Timing, Oscillations & Criticality
- **Oscillations**: Functional timing for phase separation (predict vs update), routing windows, and cross-column synchronization. Not biological decoration.
- **Criticality Control**: Maintain near-critical regime via inhibition strength, routing sparsity, gain/noise knobs. Avoid subcritical (dead) or supercritical (chaotic) states.
- **Temporal Modeling**: Explicit sequence prediction `x(t) → x(t+1)`, delay buffers, recurrent state `h`, eligibility traces.

## 17. Objective Function & Optimization
Minimize:
```
L = E_pred + λ_sparsity · C_sparsity + λ_energy · C_energy
```
Where:
- `E_pred`: sensory + latent + outcome prediction error
- `C_sparsity`: penalizes excess activity and connections (enforces top-k)
- `C_energy`: penalizes instability, runaway gain, and weight norm explosion
- Grounded in consequence-based error to turn prediction into behavior shaping.

## 18. Scaling Constraints & Computational Limits
| Constraint | Target |
|------------|--------|
| Connection density | 1–5% active connections |
| Activation sparsity | 2–10% active columns per step |
| Routing bandwidth | Top-k only (`k << N`) |
| Caps | Fan-in/fan-out per column, workspace size, memory growth (decay/consolidation) |

Prevents compute explosion, noise drift, and catastrophic forgetting.

## 19. Operational Execution Loop (Timestep Workflow)
1. **Observe**: Environment interface delivers `x_bottom`, task context, reward/modulation signals.
2. **Retrieve**: Memory layer computes keys, returns `mem_ctx` to active columns.
3. **Route**: Columns compute `s_route`; workspace aggregates, selects top-k.
4. **Infer**: Selected columns run iterative relaxation (`basal ↔ apical`), update `h`, compute `e_local`, `y_pred`, `a_prop`.
5. **Search/Reason**: Planner simulates top candidates, scores via error/reward/stability, selects best action.
6. **Act & Adapt**: Execute action, update environment, record trace in working/episodic memory.
7. **Modulate**: ACh/DA/NE adjust gain, precision, and exploration/exploit balance.
8. **Consolidate (Offline/Dream)**: Replay traces, reinforce stable patterns, decay stale engrams, update fast adapters.

## 20. Success Criteria & Validation
The system is successful if it:
- Predicts and acts coherently across timesteps.
- Improves from consequences, not just static inputs.
- Maintains stable sparse dynamics and bounded gain.
- Uses memory effectively across working, episodic, and long-term timescales.
- Adapts rapidly without catastrophic collapse or core weight destabilization.
- Scales via swarm coordination, not monolithic size.
- **Validation tests**: Convergence on stable inputs, partial-input completion, measurable mismatch reduction, hierarchical prior influence, local updates without global backprop.

## 21. Open Questions & Future Extensions
- Optimal balance between biological realism and numerical stability.
- Continuous vs. spiking compartmental signals for energy efficiency.
- Outer meta-optimizer vs. purely local/plastic learning regimes.
- Cross-hierarchical memory trace representation and routing alignment.
- **Planned extensions**: Spiking event-driven dynamics, Mamba-like selective state flow, advanced temporal gating, meta-learning adapter initialization, multi-agent swarm communication protocols.

---
*This specification unifies predictive coding attractor dynamics, compartmental inference, sparse routing, multi-timescale memory, and closed-loop action into a coherent, implementable architecture. All components are designed to operate locally, scale sparsely, and adapt continuously without reliance on global backpropagation.*