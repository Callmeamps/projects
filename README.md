# Bio-Inspired Cortical Microcircuit (Hierarchical & Tensorized)

A scalable, hierarchical implementation of the canonical cortical microcircuit.

## Architecture Highlights
- **Vectorized Core**: 3D tensors and `np.einsum` enable biological real-time simulation of 500+ columns (~50ms/step).
- **Hierarchical Depth (V1 <-> V2)**:
  - **Predictive Coding**: Higher areas predict lower area activity; lower areas process the residual (surprise).
  - **Temporal Pooling**: Higher areas integrate over longer windows (1/5th update rate), creating stable abstract representations.
- **Local Learning**: Hebbian/Anti-Hebbian plasticity with local eligibility traces. No global backprop.
- **Sensorimotor**: Integrated motor output layer (L5b) with reward-modulated (Dopamine) learning.
- **Structural Plasticity**: Dynamic pruning and regrowing of sparse lateral connections.
- **Memory Consolidation**: Generative sleep/replay cycles to stabilize synaptic knowledge.
- **Engram Memory Layer**: Fast associative memory using Modern Hopfield retrieval for patterns, skills, and episodic traces.

## Performance & Results
- **V1 -> V2 Stability**: V2 hidden state is ~10x more temporally stable than V1 on noisy pattern sequences.
- **Sensorimotor**: Agent successfully learns 1D navigation (Goal Reached in <40 steps) using local reinforcement.

## Usage
### Single Sheet
```python
from cortical_sheet_tensor import TensorizedCorticalSheet, ColumnConfig
cfg = ColumnConfig(d_in=16, d_h=32)
sheet = TensorizedCorticalSheet(n_cols=100, cfg=cfg)
res = sheet.step(input_vec)
```

### Hierarchy
```python
from cortical_hierarchy import CortexHierarchy
hier = CortexHierarchy()
hier.add_area("V1", n_cols=50)
hier.add_area("V2", n_cols=20)
hier.link_ff("V1", "V2")
hier.link_fb("V2", "V1")
out = hier.step(input_vec)
```

## Directory Structure
- `cortical_column.py`: Phase 0-1 foundation.
- `cortical_sheet_tensor.py`: Scalable sheet & motor logic.
- `cortical_hierarchy.py`: Multi-area management.
- `engram_memory.py`: Fast associative memory (Modern Hopfield retrieval).
- `test_engram_memory.py`: Tests for engram memory layer.
- `bench_nav.py`: Sensorimotor benchmark.
- `bench_hierarchy.py`: Temporal stability benchmark.
- `TODO.md`: Roadmap.

## Future (Phase 10)
- Evolutionary optimization of column hyperparameters.
- GPU acceleration via JAX/PyTorch.
