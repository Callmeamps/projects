Cortical Microcircuit: Unified Specification

A Predictive Swarm Architecture for Continuous Learning Agents

---

1. Scope & Philosophy

1.1 Purpose

This specification synthesizes two complementary designs:

· Biological Predictive Coding Cortex: compartmental neurons, local error reduction, hierarchical relaxation
· Cortical Microcircuit Expansion: swarm-based coordination, action loops, memory layers, multi-environment adaptation

The unified system is a small, continuous-learning omnimodel swarm that can predict, reason, and act without global backpropagation.

1.2 Core Principles

Principle Implication
Intelligence emerges from local prediction + local correction No single dense network; no global loss
Sparse routing by default Compute is cheap, coordination is bottlenecked
Memory is explicit, not just weights Engrams, episodes, working memory
Action closes the loop Prediction error alone is insufficient
Biology is reference, not blueprint Keep only functionally useful abstractions

1.3 Non-Goals

· Exact biological fidelity
· Giant foundation model scale
· Pure reconstruction or pure language modeling
· Global backpropagation for learning
· Hard spike simulation (unless later extended)

---

2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ENVIRONMENT INTERFACE                        │
│   (Chat / Game / Tool / Sequence / Simulated Agent)             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    THALAMIC ROUTING LAYER                        │
│             Sparse competition over column scores               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PREDICTIVE COLUMNS                          │
│   (Swarm of predictive control units, each with state + error)  │
└───────────────┬───────────────┬───────────────┬─────────────────┘
                │               │               │
                ▼               ▼               ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│ Inhibitory Control│ │   Global Workspace │ │  Engram Memory    │
│ (PV/SST/VIP)      │ │   (Top-k + Broadcast)│ │  (Fast associative)│
└───────────────────┘ └───────────────────┘ └───────────────────┘
                │               │               │
                └───────────────┴───────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              ADAPTER / PLASTICITY LAYER + PLANNER                │
│         (Task-conditioned fast adaptation + search loop)        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ACTION SELECTION (BG analogue)                │
│              Go/No-go gating over candidate actions             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                          [ENVIRONMENT]
```

2.1 Eight Main Subsystems

# Subsystem Role
1 Predictive Columns Core predictive control units with local state
2 Inhibitory Control PV/SST/VIP-style gain & switching regulation
3 Thalamic Routing Sparse competition & column selection
4 Global Workspace Bottleneck for coordination & broadcast
5 Engram Memory Fast associative retrieval & compression
6 Adapter/Plasticity Task-conditioned fast parameters
7 Planner/Search Internal rollouts & candidate evaluation
8 Action Selection Basal-ganglia-like commitment mechanism

---

3. Unified Column Specification

3.1 Column as Predictive Control Unit

Each column is no longer just a predictor — it is a predictive control unit that:

Function Description
Predict next input or next latent state
Estimate local uncertainty/precision
Propose candidate action or interpretation
Integrate bottom-up, top-down, lateral context
Emit routing score and memory keys
Maintain short-term recurrent state

3.2 Column Inputs

Signal Source Description
x_bottom Sensory/feature input Feedforward evidence
x_top Higher column / workspace Top-down context/prior
x_lateral Neighbor columns Sparse lateral coordination
r_route Thalamic routing Gate/attention signal
m_global Neuromodulators ACh/DA/NE state
w_task Task encoder Mode token (chat/game/plan)
mem_ctx Engram memory Retrieved episodic trace

3.3 Column Internal State

Variable Role
h Recurrent latent state
p Precision / confidence estimate
g Inhibitory gain (PV-controlled)
e_trace Eligibility trace for credit assignment
u Action proposal state
k Workspace / retrieval key

3.4 Column Outputs

Output Destination
y_pred Next input/latent prediction → error computation
e_local Local prediction error → routing & learning
a_prop Proposed action/symbol → action selection
s_route Routing score → thalamic competition
h_next Updated latent → recurrent state
k_mem Memory key → engram retrieval
v_mem Compressed value → engram storage

---

4. Compartment Model (Pyramidal Neuron Abstraction)

4.1 Three-Compartment Architecture

```
                     ┌─────────────────┐
                     │   APICAL (top)   │ ← Top-down expectation
                     │   Context bias   │
                     └────────┬────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         ▼                         │
    │                    ┌─────────┐                    │
    │                    │  SOMA   │ ← Integration       │
    │                    │ Output  │   & decision        │
    │                    └────┬────┘                    │
    │                         │                         │
    │    ┌────────────────────┼────────────────────┐    │
    │    │                    │                    │    │
    │    ▼                    ▼                    ▼    │
    │ ┌─────────┐      ┌───────────┐      ┌───────────┐ │
    │ │ BASAL   │      │ LOCAL     │      │ ACTION    │ │
    │ │ (bottom)│      │ ERROR     │      │ PROPOSAL  │ │
    │ └─────────┘      └───────────┘      └───────────┘ │
    │                                                    │
    └────────────────────────────────────────────────────┘
```

4.2 Compartment Roles

Compartment Function Biological Analogue
Basal Encodes feedforward evidence, drives local prediction Basal dendrites
Apical Encodes top-down expectation, modulates confidence & gain Apical dendrites
Soma Integrates both streams, produces active representation, emits action proposal Pyramidal soma

4.3 Mathematical Formulation

Let for neuron/column i:

· b_i = basal drive (feedforward evidence)
· a_i = apical drive (top-down expectation)
· s_i = soma state (current representation)

Local mismatch:

```
ε_i = b_i - a_i
```

State update (continuous-time relaxation):

```
s_i^{t+1} = s_i^t + η (α·b_i + β·a_i + γ·ε_i)
```

where η = step size, α,β,γ = compartment weights.

Firing condition: Output depends on alignment between evidence and expectation, but remains useful for action selection.

4.4 Energy Interpretation

Total energy minimized during inference:

```
E = ½ Σ_i ||b_i - a_i||²
```

Settled state is a local minimum of this energy — functionally equivalent to attractor dynamics.

---

5. Inhibition & Stability (PV/SST/VIP)

The inhibitory system is first-class, not an afterthought.

5.1 Interneuron Types

Type Target Effect When Active
PV (Parvalbumin) Soma/perisomatic of excitatory Fast divisive gain control, top-k sparsity High input drive, runaway risk
SST (Somatostatin) Apical dendrites Suppress top-down influence Noisy/unhelpful context, prevent hallucination
VIP (Vasoactive Intestinal Peptide) PV/SST interneurons Disinhibit columns (VIP → PV/SST → excite) Exploration, switching, novelty

5.2 Stability Targets

Constraint Mechanism
Sparse activity by default PV-mediated top-k competition
No runaway firing Divisive gain normalization
Bounded gain Clipping + homeostatic regulation
Graceful explore/exploit switching VIP disinhibition + NE modulation

5.3 Mathematical Form

Column j receives effective inhibition:

```
I_j = PV_j · s_j + SST_j · apical_j
```

where PV_j and SST_j are interneuron outputs.

VIP disinhibition:

```
PV_effective = PV_base - VIP_influence
```

---

6. Routing & Workspace

6.1 Routing Goal

Decide which columns compute, communicate, and influence the workspace using sparse competition.

6.2 Routing Score Composition

Each column emits a routing score s_route derived from:

· Prediction error (low error → high score)
· Novelty (unexpected patterns)
· Task relevance (task vector similarity)
· Reward/usefulness (recent contribution to reward)
· Confidence/precision (p variable)
· Memory match strength (engram retrieval similarity)

6.3 Thalamic Competition (Top-k Routing)

```
active_columns = top_k({s_route_i}, k = κ·N)
```

where κ ∈ [0.02, 0.10] (2–10% activation sparsity).

6.4 Global Workspace

A small shared bottleneck that:

· Collects top-k column outputs
· Holds task context and mode token
· Broadcasts compact state back to columns
· Acts as coordination hub for thought

Workspace contents:

```
workspace = {
    "active_reps": [h_i for i in active],
    "action_candidates": [a_prop_i for i in active],
    "task_mode": one_hot({chat, game, plan, explore}),
    "broadcast": compact_state (dim << N),
    "retrieved_mem": engram lookup result
}
```

6.5 Workspace Function

The workspace is where:

· Multiple columns unify their proposals
· Action candidates compete via BG analogue
· Memory retrieval is consulted
· Task mode is stabilized across timesteps

This is the bridge between local computation and global behavior.

---

7. Memory System

Weights alone are insufficient. Memory must be explicit.

7.1 Memory Components

Component Timescale Update Role
Working memory Step + short traces Immediate Current state buffer
Episodic memory Task events/episodes Rapid, decay Temporary sequence storage
Engram memory Long-term Reinforce/decay Compressed patterns, skills
Replay buffer Offline Sample + reinforce Consolidation

7.2 Engram Memory (Fast Associative)

Storage: (key, value, age, strength)

· key: sparse activity pattern (e.g., top-k column keys)
· value: compressed representation or action trace
· strength: reinforced when useful, decays when unused

Retrieval:

```
key_current = compress(active_columns + workspace)
matches = top_m(cosine_sim(key_current, key_stored))
retrieved = weighted_sum(value_matches)
```

Lifecycle:

· Reinforce when retrieval reduces prediction error or leads to reward
· Decay via exponential decay on unused engrams
· Consolidate when repeatedly confirmed (move to slow memory)
· Maintain compressibility via sparsity constraints

7.3 Hippocampal-Cortical Analogue (CLS-style)

System Role Plasticity
Fast (hippocampal) Rapid episodic storage High, temporary
Slow (cortical) Gradual integration Low, stable
Replay bridge Offline pattern transfer Consolidation

---

8. Action Loop (Closing the System)

Prediction alone is shallow. The system must act and observe consequences.

8.1 Action Outputs (per timestep)

Action Type Example
Tool call search(query), calculate(expr)
Chat token Response chunk
Game action move(dx,dy), attack(target)
Memory operation retrieve(key), store(pattern)
No-op When confidence < threshold

8.2 Action Scoring (for candidate evaluation)

Score each candidate action using:

```
score(a) = w_pred·Δerror + w_reward·E[reward] + w_novelty·novelty + w_task·task_match - w_risk·stability_risk
```

8.3 Basal Ganglia Analogue (Selection)

Separate generation from selection:

Component Function
Direct pathway (Go) For high-scoring actions, facilitate execution
Indirect pathway (No-go) Suppress low-scoring or risky actions
STN Global inhibition, reset

Selection rule:

```
selected_action = argmax_{a in candidates} (Go(a) - NoGo(a))
```

Only commit if Go - NoGo > threshold.

8.4 Closed-Loop Dynamics

```
Action → Environment → New Observation → Prediction Update → Memory Update
```

Error definition now includes both:

```
E_total = E_sensory_prediction + E_outcome (expected vs actual result)
```

---

9. Learning & Credit Assignment

Pure Hebbian is insufficient. Use a hybrid local learning regime.

9.1 Learning Mechanisms

Mechanism Description
Local Hebbian ΔW ∝ pre·post (fast, unsupervised)
Error-modulated ΔW ∝ ε·pre (local prediction error)
Eligibility traces Temporal bridge for delayed outcomes
Proxy gradients Local difference signals between layers

9.2 Column Learning Update

For column i:

```
ΔW_basal,i = η_local · ε_i · h_bottom
ΔW_apical,i = η_local · ε_i · h_top
```

Eligibility trace maintains credit over multiple steps:

```
e_trace_i^{t+1} = λ·e_trace_i^t + ∂s_i/∂W_i
ΔW_i = η · δ_reward · e_trace_i
```

9.3 No Global Backpropagation

The system achieves deep coordination through:

· Hierarchical prediction error propagation
· Layer-local error targets
· Sparse routing of error signals
· Local learning rules with temporal eligibility

---

10. Reward & Neuromodulation

Prediction error is insufficient. Add weak reward-like signals.

10.1 Reward Sources

Source When
Task success Goal achieved
User approval Explicit feedback, satisfaction
Environment progress Subgoal completion
Reduced uncertainty Confidence increase
Memory usefulness Successful retrieval

10.2 Neuromodulator Mapping

Modulator Role Effect
ACh (Acetylcholine) Sharpen sensory gain, attention Increase basal precision
DA (Dopamine) Reinforce useful actions & memory Strengthen selected pathways
NE (Norepinephrine) Reset, exploration, mode switch Trigger volatility, increase noise

10.3 Mode-Dependent Behavior

The same network behaves differently based on modulatory state:

Mode ACh DA NE Behavior
Passive chat Low Low Low Default, predictive
Active reasoning High Med Low Deep inference
Exploration Low Low High Novel actions, VIP disinhibition
Planning Med Med Med Internal search rollouts
Tool use High Med Med Action-focused
Game control Med High Low Reward-chasing

---

11. Planning & Internal Search

Add a small search loop for deliberate reasoning.

11.1 Search Process (per decision step)

```
1. PROPOSE: Generate k candidate actions/continuations
2. SIMULATE: Rollout each candidate using predictive core (depth d)
3. SCORE: Evaluate predicted error, reward, stability
4. SELECT: Choose best candidate via BG analogue
```

11.2 Implementation Practicalities

Technique Application
Short rollouts 3–5 step lookahead
Beam search Width = 3–5 candidates
Branch-and-score Depth-first with pruning
Self-check Task-specific verification pass

11.3 When to Search

· Uncertainty > threshold
· Task mode = "planning" or "reasoning"
· Low confidence in first-choice action
· Explicit instruction (e.g., "think step by step")

---

12. Swarm Structure & Specialization

The full system is a swarm of smaller units, not one giant circuit.

12.1 Specialization Through Use

Sheets/tiles/clusters may specialize in:

Specialization Function
Language Chat, instruction following
Memory Engram storage/retrieval
Planning Search, rollouts
Perception Sensory encoding
Game control Action mapping
Tool use API coordination
Long-horizon state Episodic tracking

Specialization emerges from use patterns, not hard-coded roles.

12.2 Communication Constraints

Constraint Value
Sparse ≤10% columns communicate per step
Bottlenecked Through global workspace
Bandwidth-limited Workspace dimension << total state
Memory-mediated Retrieval can bypass direct wiring

12.3 Scaling Constraints (Enforced)

Constraint Target
Connection density 1–5% active connections
Activation sparsity 2–10% active columns per step
Routing bandwidth Top-k only, k << N
Workspace size Fixed small dimension
Memory growth Decay + consolidation limits

Without these, the system explodes in compute and noise.

---

13. Hierarchy & Temporal Modeling

13.1 Cortical Hierarchy Layers

Layer Role Signal direction
L4-like Primary input encoding Bottom-up from sensory
L2/3-like Integration, lateral mixing Within-area
L5-like Action/output projection Bottom-up to subcortex
L6-like Feedback, routing influence Top-down modulation

13.2 Pathways

Pathway Direction Role
Feedforward Bottom → up Evidence propagation
Feedback Top → down Context priors
Lateral Same level Coordination, competition

13.3 Temporal Mechanisms

Mechanism Purpose
Delay buffers Short queues for sequence
Recurrent state h History compression
Eligibility traces Credit over time
Sequence prediction x(t) → x(t+1)

13.4 Oscillations (Functional Timing)

Frequency Phase Function
Gamma/theta Predict phase Bottom-up drive
Beta Update phase Error correction
Delta Reset Mode switching

Oscillations are functional scheduling mechanisms, not biological decoration.

13.5 Criticality Control

Maintain near-critical regime to avoid:

· Subcritical: Dead activity, no propagation
· Supercritical: Chaos, runaway excitation

Control knobs:

· Inhibition strength (PV gain)
· Routing sparsity (top-k ratio)
· Noise injection (NE-modulated)

---

14. Environment Interface

14.1 Shared Environment API

```python
class Environment:
    def step(self, action) -> (observation, reward, done, info)
    def reset(self) -> observation
    def action_space(self) -> Space
    def observation_space(self) -> Space
```

14.2 Supported Environment Categories

Category Inputs Outputs Use Case
Chat User text, conversation state, tool feedback Response, tool calls, queries Dialogue, instruction following
Game State, map, reward, inventory Movement, attack, build, select Control, planning
Simulated agent Task description, observations, tool output Tool actions, plans, status Autonomy, coordination
Sensory/sequence Patches, time series, symbols Next prediction Toy training, stability tests

---

15. Objective Functions

15.1 Total Objective

```
L = E_prediction + λ_sparsity·L_sparsity + λ_energy·L_energy + λ_reward·L_reward
```

15.2 Component Definitions

Prediction error:

```
E_pred = Σ_t (||x_pred(t) - x_obs(t)||² + ||outcome_pred(t) - outcome_actual(t)||²)
```

Sparsity cost:

```
L_sparsity = Σ_i |activity_i| + Σ_{i,j} |W_{ij}|   (with top-k enforcement)
```

Energy/stability cost:

```
L_energy = Σ_i (gain_i² + ||s_i||²) + penalty for runaway firing
```

Reward objective (optional):

```
L_reward = -Σ_t reward(t)   (maximize cumulative reward)
```

---

16. Dream, Replay & Consolidation

16.1 Dream Mode (Offline)

Recombine stored patterns to generalize structure:

```
dream_pattern = mix(engram1, engram2, noise)
run_inference(dream_pattern)
reinforce stable attractors
```

16.2 Nightmare Mode

Replay failures to stress-test predictions:

```
replay_bad_episodes()
identify_weak_predictions()
generate_adversarial_variations()
```

16.3 Replay Schedule

Phase Frequency Content
Immediate Every step Working memory flush
Episodic Episode end Full trajectory
Consolidation Periodic (sleep-like) Sampled engrams

---

17. Task Contracts & Success Criteria

17.1 Task Contract Definition

Each task must define:

Element Description
Win condition Goal achievement signal
Lose condition Failure, timeout, violation
Timeout Maximum steps
Confidence threshold Minimum certainty for action

17.2 Success Criteria

The system is successful if it:

· ✅ Predicts and acts coherently in at least 3 environment types
· ✅ Improves from consequences, not just inputs
· ✅ Maintains stable sparse dynamics (2–10% activation)
· ✅ Uses memory to avoid relearning patterns
· ✅ Adapts to new tasks without collapse
· ✅ Scales via swarm size, not parameter explosion

---

18. API Specification

18.1 Column API

```python
class PredictiveColumn:
    def forward(self, 
                x_bottom: Tensor,      # sensory input
                x_top: Optional[Tensor], # top-down context
                x_lateral: Optional[Tensor], # lateral from neighbors
                r_route: Optional[Tensor], # routing gate
                m_global: Dict[str, float], # neuromodulators
                w_task: Optional[Tensor], # task vector
                mem_ctx: Optional[Tensor]  # retrieved memory
               ) -> ColumnOutput:
        """Update column state and return outputs."""
```

18.2 System API

```python
class CorticalSwarm:
    def step(self, observation: Observation) -> Action:
        """One environment step: infer → select → act → learn."""
    
    def reset(self, task_contract: TaskContract) -> None:
        """Reset state for new episode."""
    
    def train(self, experience: Experience) -> None:
        """Online learning from step outcome."""
    
    def dream(self, steps: int = 100) -> None:
        """Offline consolidation."""
    
    def set_mode(self, mode: Mode) -> None:
        """Switch between chat/game/plan/explore."""
```

18.3 Memory API

```python
class EngramMemory:
    def retrieve(self, key: Tensor, k: int = 5) -> List[MemoryTrace]:
        """Return top-k matching memory traces."""
    
    def store(self, key: Tensor, value: Tensor, strength: float = 1.0) -> None:
        """Store or reinforce engram."""
    
    def decay(self, rate: float = 0.01) -> None:
        """Age and prune weak engrams."""
    
    def consolidate(self) -> None:
        """Transfer reinforced patterns to slow memory."""
```

---

19. Reference Implementation Sketch (PyTorch)

```python
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class ColumnOutput:
    y_pred: torch.Tensor
    e_local: torch.Tensor
    a_prop: torch.Tensor
    s_route: float
    h_next: torch.Tensor
    k_mem: torch.Tensor
    v_mem: torch.Tensor

class PyramidalCompartment(nn.Module):
    """Single pyramidal neuron with basal/apical/soma."""
    def __init__(self, dim: int):
        super().__init__()
        self.W_basal = nn.Linear(dim, dim, bias=False)
        self.W_apical = nn.Linear(dim, dim, bias=False)
        self.soma = torch.zeros(dim)
        self.p = 1.0  # precision
        self.g = 1.0  # gain
    
    def forward(self, x_bottom: torch.Tensor, x_top: torch.Tensor, 
                lr: float = 0.1) -> torch.Tensor:
        basal = self.W_basal(x_bottom)
        apical = self.W_apical(x_top)
        error = basal - apical
        self.soma = self.soma + lr * (basal + apical + error)
        return self.soma

class PredictiveColumn(nn.Module):
    def __init__(self, dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([PyramidalCompartment(dim) for _ in range(num_layers)])
        self.h = torch.zeros(dim)
        self.p = 1.0
        self.g = 1.0
        self.e_trace = torch.zeros(dim)
        self.u = torch.zeros(dim)
    
    def forward(self, x_bottom: torch.Tensor, x_top: Optional[torch.Tensor] = None,
                x_lateral: Optional[torch.Tensor] = None, r_route: float = 1.0,
                m_global: Optional[Dict[str, float]] = None,
                w_task: Optional[torch.Tensor] = None,
                mem_ctx: Optional[torch.Tensor] = None,
                steps: int = 5) -> ColumnOutput:
        
        state = self.h
        if x_top is None:
            x_top = state
        if mem_ctx is not None:
            state = state + 0.1 * mem_ctx
        
        # Iterative inference
        for _ in range(steps):
            for i, layer in enumerate(self.layers):
                x_b = state if i == 0 else self.layers[i-1].soma
                x_t = x_top if i == len(self.layers)-1 else self.layers[i+1].soma
                state = layer(x_b, x_t)
        
        # Prediction
        y_pred = self.layers[-1].soma
        
        # Local error
        e_local = (y_pred - x_bottom).norm()
        
        # Routing score
        s_route = (1.0 / (1.0 + e_local)) * self.p * r_route
        
        # Action proposal (simple: from top layer)
        a_prop = torch.tanh(self.layers[-1].soma.mean())
        
        self.h = state
        return ColumnOutput(
            y_pred=y_pred, e_local=e_local, a_prop=a_prop,
            s_route=s_route, h_next=state, 
            k_mem=state, v_mem=self.layers[-1].soma
        )

class ThalamicRouter(nn.Module):
    def __init__(self, num_columns: int, sparsity: float = 0.05):
        super().__init__()
        self.num_columns = num_columns
        self.sparsity = sparsity
        self.k = max(1, int(num_columns * sparsity))
    
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        # Top-k selection
        top_k_values, top_k_indices = torch.topk(scores, self.k)
        mask = torch.zeros_like(scores)
        mask[top_k_indices] = 1.0
        return mask

class GlobalWorkspace(nn.Module):
    def __init__(self, dim: int, capacity: int = 32):
        super().__init__()
        self.capacity = capacity
        self.context = torch.zeros(dim)
        self.task_mode = torch.zeros(4)  # chat, game, plan, explore
        
    def aggregate(self, active_reps: List[torch.Tensor]) -> torch.Tensor:
        if not active_reps:
            return self.context
        # Simple averaging (could be attention)
        self.context = torch.stack(active_reps[:self.capacity]).mean(dim=0)
        return self.context

class EngramMemory:
    def __init__(self, max_size: int = 1000, decay_rate: float = 0.01):
        self.keys = []
        self.values = []
        self.strengths = []
        self.max_size = max_size
        self.decay_rate = decay_rate
    
    def retrieve(self, key: torch.Tensor, k: int = 5) -> List[torch.Tensor]:
        if not self.keys:
            return []
        # Cosine similarity
        sims = [torch.cosine_similarity(key, k, dim=0).item() for k in self.keys]
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [self.values[i] for i in top_indices if sims[i] > 0.5]
    
    def store(self, key: torch.Tensor, value: torch.Tensor, strength: float = 1.0):
        self.keys.append(key.detach().clone())
        self.values.append(value.detach().clone())
        self.strengths.append(strength)
        if len(self.keys) > self.max_size:
            # Decay weakest
            min_idx = torch.argmin(torch.tensor(self.strengths)).item()
            del self.keys[min_idx], self.values[min_idx], self.strengths[min_idx]

class CorticalSwarm(nn.Module):
    def __init__(self, num_columns: int = 64, column_dim: int = 128):
        super().__init__()
        self.columns = nn.ModuleList([PredictiveColumn(column_dim) for _ in range(num_columns)])
        self.router = ThalamicRouter(num_columns)
        self.workspace = GlobalWorkspace(column_dim)
        self.memory = EngramMemory()
        self.history = []  # for replay
        
    def step(self, observation: torch.Tensor, task_mode: str = "chat") -> torch.Tensor:
        # Map task mode to vector
        mode_vec = torch.zeros(4)
        mode_map = {"chat": 0, "game": 1, "plan": 2, "explore": 3}
        mode_vec[mode_map[task_mode]] = 1.0
        self.workspace.task_mode = mode_vec
        
        # 1. Compute column outputs
        outputs = []
        for col in self.columns:
            mem = self.memory.retrieve(col.h)
            mem_ctx = mem[0] if mem else None
            out = col(observation, x_top=self.workspace.context, 
                      w_task=mode_vec, mem_ctx=mem_ctx)
            outputs.append(out)
        
        # 2. Routing (sparse activation)
        scores = torch.tensor([o.s_route for o in outputs])
        mask = self.router(scores)
        active_indices = torch.where(mask > 0)[0]
        
        # 3. Workspace aggregation
        active_reps = [outputs[i].h_next for i in active_indices]
        self.workspace.aggregate(active_reps)
        
        # 4. Action selection (simplified)
        action_proposals = torch.stack([outputs[i].a_prop for i in active_indices])
        if len(action_proposals) > 0:
            action = action_proposals.mean()  # placeholder
        else:
            action = torch.zeros(1)
        
        # 5. Memory update (store trace)
        for i in active_indices:
            self.memory.store(outputs[i].k_mem, outputs[i].v_mem)
        
        # 6. Store experience for replay
        self.history.append((observation, action, outputs))
        
        return action
    
    def learn_from_reward(self, reward: float):
        """Simple reward-modulated learning."""
        for col in self.columns:
            # Reward-modulated Hebbian (placeholder)
            pass
```

---

20. Open Questions & Future Extensions

Question Status Notes
Balance of biological realism vs stability Ongoing Start abstract, add detail as needed
Continuous vs spiking compartment signals Deferred Continuous for initial implementation
Purely local vs hybrid learning Hybrid Local error + neuromodulated
Memory consolidation trigger Open Experience-dependent or periodic?
Hierarchical depth Configured 3–5 layers typical
Action space granularity Environment-dependent Abstract actions first

Approved Extensions

· ✅ Modern Hopfield retrieval for engrams
· ✅ Mamba-like state flow for sequences
· ✅ Spiking dynamics (if needed later)
· ✅ Temporal gating for multi-timescale

---

21. Summary

This unified specification describes a small, continuous-learning omnimodel swarm that:

1. Predicts using local column-level inference with compartmental neurons
2. Routes computation sparsely via top-k thalamic competition
3. Coordinates through a small global workspace bottleneck
4. Remembers via explicit engram, episodic, and working memory
5. Acts through closed-loop action selection with reward modulation
6. Learns locally without global backpropagation
7. Adapts via fast task-conditioned plastic modules
8. Reasons using short internal search rollouts
9. Consolidates via replay, dream, and nightmare modes
10. Scales through swarm specialization, not parameter explosion

The target is not a giant general model but a coordinated system of predictive units that can operate in chat, game, tool use, and simulated environments.