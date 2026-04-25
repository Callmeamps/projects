# Bio-Inspired Cortical Microcircuit (Tensorized)

Vectorized implementation of a cortical sheet based on canonical microcircuit principles (Basal/Apical separation, local plasticity, inhibitory control).

## Core Architecture
- **Tensorized Forward Pass**: 3D Weight Tensors `[n_cols, d_out, d_in]`. Vectorized using `np.einsum`.
- **Local Plasticity**: Hebbian/Anti-Hebbian updates modulated by local prediction error and global Dopamine.
- **Inhibitory Control**: PV (gain), SST (apical gating), and VIP (disinhibition) dynamics.
- **Structural Plasticity**: Synaptic pruning and probabilistic growth for small-world connectivity.
- **Memory Consolidation**: Generative sleep/replay cycles to stabilize synaptic tags into long-term weights.
- **Sensorimotor**: L5b motor layer with dopamine-modulated action selection.

## Performance
Scales linearly with column count:
- 100 Columns: ~10ms per step
- 500 Columns: ~55ms per step

## Directory Structure
- `cortical_column.py`: Phase 0-1 foundation (Config, Neuromod).
- `cortical_sheet_tensor.py`: Phase 7-8 implementation (Sheet-level tensors).
- `bench_nav.py`: Sensorimotor verification task (1D Navigation).
- `TODO.md`: Project roadmap and phase status.

## Usage
```python
from cortical_sheet_tensor import TensorizedCorticalSheet, ColumnConfig

cfg = ColumnConfig(d_in=16, d_h=32)
sheet = TensorizedCorticalSheet(n_cols=100, cfg=cfg)

# Step
res = sheet.step(input_vector, reward=1.0)
print(f"Action: {res['action']}")

# Consolidate
sheet.sleep_step(n_steps=50)
```

## Status
- **Phase 1-6**: OO-Foundation (Completed)
- **Phase 7**: Tensorization & Scaling (Completed)
- **Phase 8**: Sensorimotor Integration (Completed)
- **Phase 9**: Multi-Area Hierarchy (Planned)
