# Spec v0.4

1. Goal

Build a minimal cortical system that implements predictive coding using modular cortical columns with:

compartmentalized pyramidal neurons (basal/apical/soma)

inhibitory interneuron control (PV/SST/VIP)

local error-driven plasticity

thalamus-like routing

neuromodulatory state gating

sparse recurrent column network


The system should learn online, remain stable, and support context switching without global backpropagation.


---

2. Core Design Principle

> Intelligence emerges from distributed prediction + local error correction + gated plasticity + sparse routing



Not:

global gradient descent

monolithic networks


But:

many small predictive units cooperating under constraints



---

3. Column Specification

3.1 Inputs

Each column receives:

x_bottom: sensory / feature input

x_top: contextual / top-down prediction

x_lateral: neighbor column activity

r_thalamic: routing signal (attention / gating)

m_global: neuromodulatory state vector



---

3.2 Internal State

h: recurrent latent state

p: prediction confidence (precision estimate)

g: inhibitory gain state

e_trace: error eligibility memory



---

3.3 Outputs

y_pred: predicted next input

e_local: prediction error

route_score: importance / salience

h_next: updated state



---

4. Neuron Model (Pyramidal Unit)

Each neuron has 3 compartments:

Basal dendrites

encode feedforward evidence

drives initial activation


Apical dendrites

encode top-down context

modulates gain / expectation


Soma

integrates basal + apical

produces spike / activation


Key rule:

> firing depends on alignment between evidence (basal) and expectation (apical)




---

5. Inhibitory System

PV interneurons

fast stabilization

controls gain and synchrony

prevents runaway activation


SST interneurons

suppress apical influence

stabilizes context integration


VIP interneurons

disinhibition mechanism

enables switching / exploration modes



---

6. Routing (Thalamus-like System)

Functions:

selects active columns per timestep

gates communication bandwidth

amplifies high-salience predictions

suppresses irrelevant activity


Mechanism:

competition-based scoring over columns

sparse activation (top-k or thresholded routing)



---

7. Plasticity System

Hebbian

Strengthen correlated activity: Δw ∝ x · y

Anti-Hebbian

Encourages decorrelation and sparsity

STDP-like

Temporal causality learning:

pre-before-post strengthens

post-before-pre weakens


Error-modulated learning

Weight updates scaled by prediction error magnitude

Gated plasticity

Plasticity depends on neuromodulatory state:

learning enabled only under appropriate conditions



---

8. Neuromodulatory State

Global vector controlling system mode:

ACh: sensory gain / attention sharpening

DA: learning salience / reward weighting

NE: uncertainty / reset / exploration trigger


Effect:

> same circuit behaves differently under different global states




---

9. Timing System

Simplified phase-based operation:

1. Feedforward phase


2. Prediction phase


3. Error computation phase


4. Inhibition / competition phase


5. Plasticity update phase


6. State propagation phase



Purpose:

separates inference from learning

prevents interference between update types



---

10. Network Structure

columns arranged in sparse graph (grid or small-world)

local connectivity dominates

long-range links are rare and learned

shared weights possible across column types



---

11. Data Flow Loop

Each timestep:

1. Receive inputs (x_bottom, x_top, x_lateral)


2. Generate prediction y_pred


3. Compute error e_local = x_bottom - y_pred


4. Update inhibitory state (PV/SST/VIP)


5. Compute routing scores


6. Select active columns (thalamic gating)


7. Apply plasticity updates (error + eligibility traces)


8. Update recurrent state h


9. Propagate state forward




---

12. Expanded Build Order (Detailed)
### Phase 0: Foundations and Constraints

**Goal**: Define the interfaces, timescales, and stability rules so local learning remains stable at scale without global backprop.

**0.1 Tensor shapes and interfaces**
- x_bottom: [batch, d_in] sensory evidence vector
- x_top: [batch, d_ctx] contextual prediction vector
- x_lateral: [batch, k_neighbors, d_lat] sparse neighbor activity
- h: [batch, d_h] private recurrent state per column
- p: precision estimate
- e_local: [batch, d_in] vector error, not scalar, computed as x_bottom minus y_pred
- y_pred: [batch, d_in]
- route_score:
- m_global: [DA][NE] with separate timescales[batch][1][ACh]

**0.2 Neuron compartment rule**
- Basal dendrites integrate feedforward evidence linearly
- Apical dendrites provide context via a multiplicative gain gate, not additive bias
- Soma output equals basal activation times sigmoid(apical), implementing coincidence detection between evidence and expectation
- This aligns with dendritic predictive coding where error units are computed locally in dendritic compartments, linking hierarchical predictive coding to balanced networks with lateral inhibition

**0.3 Inhibitory system specification**
- PV interneurons provide fast divisive gain control on soma, preventing runaway activation
- SST interneurons suppress apical influence at distal dendrites, stabilizing context integration
- VIP interneurons provide disinhibition by suppressing PV and SST, enabling switching and exploration
- Experimental data show VIP stimulation produces divisive gain modulation in inhibited populations and additive baseline shifts in delayed activated neurons, VIP cells selectively inhibit other inhibitory neurons for disinhibitory control, and VIP mediated suppression of other interneurons enhances excitatory activity through reinforcement signals

**0.4 Routing stability**
- route_score updates are low pass filtered with tau_route equals 5 timesteps to prevent chatter
- Top-k selection uses a soft threshold plus hysteresis, keep active if score stays within 10 percent of cutoff
- Feedback from e_local to routing is scaled by p to avoid amplifying noise

**0.5 Neuromodulatory timescales**
- DA: phasic bursts on 100 to 200 ms, tonic baseline on seconds, bursts increase D1 receptor occupancy and pauses decrease it
- NE: phasic locus coeruleus activation produces cortex wide NE increase that acts as an internal interrupt for cognitive shifts and network reorganization
- ACh: fast layer specific modulation of sensory gain, modeled as per column scalar updated at 20 to 50 ms
- Global vector m_global is therefore three timescale, not single step

**0.6 Plasticity grounding**
- All weight updates use local pre times post Hebbian terms scaled by e_local magnitude
- Error modulated learning uses vector e_local to gate, preserving directionality
- Eligibility traces e_trace decay with tau_elig equals 200 ms to bridge pre before post timing without backprop
- Predictive coding can approximate backprop on arbitrary graphs using local rules, supporting this design choice

**0.7 Timing and phase budget**
- Six phases are logical, not necessarily serial hardware steps
- Inference mode fuses phases 1 to 3 into one pass, learning mode keeps separation
- Maximum per timestep budget is 6 multiply accumulates per synapse, enforced to keep Phase 7 scaling feasible

**0.8 Numeric stability rules**
- PV gain is clipped to [0.2, 5.0]
- SST suppression is clipped to
- Weight norms are homeostatically scaled to target firing rate 0.05 spikes per timestep
- Anti Hebbian term strength equals 0.1 times Hebbian term to enforce sparsity[0][1]

**0.9 Phase 0 acceptance tests**
- Shape test: single column forward pass produces y_pred with correct dimensions and no NaNs for 10k random inputs
- Apical gating test: identical x_bottom with two different x_top values produces cosine similarity of y_pred less than 0.3
- Inhibition test: step input of magnitude 10x normal does not increase firing rate beyond 2x baseline after PV engages
- Routing test: with 20 columns and fixed inputs, active set changes less than 10 percent across 100 timesteps
- Neuromodulation test: same weights under high ACh versus high NE produce at least 30 percent difference in learning rate and at least 20 percent difference in PV gain.

---
Phase 1: Minimal predictive unit

Goal: validate local prediction learning

Steps:

1. Implement single column module


2. Use simple vector input/output


3. Add basal-only predictive model


4. Compute prediction error


5. Apply Hebbian + error-scaled updates


6. Train on simple sequence prediction (toy dataset)



Success criteria:

reduces prediction error over time

stable online learning



---

Phase 2: Dendritic separation

Goal: introduce biological signal separation

Steps:

1. Split input into basal vs apical streams


2. Add apical modulation of prediction


3. Implement gating effect (apical influences gain, not raw output)


4. Add simple recurrent state h


5. Test context-dependent prediction shifts



Success criteria:

same input produces different predictions under different apical context



---

Phase 3: Inhibitory control layer

Goal: stabilize and sparsify activity

Steps:

1. Add PV-style gain control (global inhibition scalar)


2. Add SST suppression of apical influence


3. Add VIP-based disinhibition switch


4. Introduce sparsity constraint on activations


5. Test stability under noisy inputs



Success criteria:

no runaway activations

sparse activation patterns emerge



---

Phase 4: Multi-column system

Goal: spatial computation + competition

Steps:

1. Create 5–20 columns with shared architecture


2. Add lateral connections between neighbors


3. Implement competition (winner-take-most or softmax routing)


4. Add local message passing between columns


5. Train on distributed feature tasks (image patches / sequences)



Success criteria:

specialization across columns emerges

redundancy decreases



---

Phase 5: Routing / thalamus layer

Goal: dynamic selection of computation

Steps:

1. Implement routing module scoring each column


2. Select top-k active columns per timestep


3. Gate communication based on routing decisions


4. Add feedback from columns to routing scores (error-driven attention)



Success criteria:

system activates different subsets depending on input

efficient sparse computation



---

Phase 6: Neuromodulatory control

Goal: context-dependent learning modes

Steps:

1. Add global state vector (ACh, DA, NE)


2. Modulate learning rate based on state


3. Modulate inhibition strength per state


4. Test task switching (same input, different objectives)



Success criteria:

system changes behavior without retraining



---

Phase 7: Full recurrent cortex tile

Goal: scalable cortical sheet

Steps:

1. Expand to 100+ columns


2. Enforce sparse connectivity constraints


3. Add long-range learned connections


4. Introduce multi-timescale plasticity (fast/slow/stable)


5. Add memory persistence via recurrent loops



Success criteria:

continual learning without catastrophic forgetting

stable long-term representations



---

13. Final System Definition

A working cortex-like system is:

> a sparse network of predictive columns using compartmentalized neurons, inhibitory gating, thalamic routing, and neuromodulator-controlled local learning rules




---

14. Key Constraint Summary

no global backprop required

no monolithic subnet per neuron

all learning is local or gated

all routing is sparse and competitive

all computation is predictive