# Cortical Microcircuit API Spec v2.0

## Overview
Predictive swarm architecture for continuous learning. Local prediction, sparse routing, no global backprop.

## Core Classes

### `ColumnConfig` (`cortical_column.py`)
Configuration for a single cortical column.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `d_in` | int | 16 | Input dimensionality |
| `d_ctx` | int | 16 | Context (x_top) dimensionality |
| `d_lat` | int | 16 | Lateral input dimensionality |
| `d_h` | int | 32 | Recurrent state size |
| `k_neighbors` | int | 4 | Lateral connections |
| `lr_base` | float | 0.01 | Learning rate |
| `tau_elig` | float | 20 | Eligibility trace decay |
| `anti_hebb_scale` | float | 0.1 | Anti-Hebbian scale |
| `target_firing_rate` | float | 0.05 | Homeostatic target |

### `TensorizedCorticalSheet` (`cortical_sheet_tensor.py`)
Main swarm of predictive columns.

#### `__init__(n_cols: int = 100, cfg: ColumnConfig = None)`
Create sheet with `n_cols` columns.

#### `step(x_bottom, x_top=None, reward=0.0, learn=True, neuromod_override=None) -> dict`
Run one step of inference + learning.

**Inputs:**
- `x_bottom` (np.ndarray): Bottom-up input, shape `(d_in,)` or `(1, d_in)` or `(n_cols, d_in)`
- `x_top` (np.ndarray, optional): Context input, shape `(d_ctx,)` or `(n_cols, d_ctx)`
- `reward` (float): Reward signal for neuromodulation
- `learn` (bool): Whether to update weights
- `neuromod_override` (Neuromodulator): Override neuromodulatory state

**Returns:** dict with keys:
- `"mean_e_mag"` (float): Mean prediction error magnitude
- `"action"` (np.ndarray): Action proposal, shape `(d_action,)`

### Environment Interface
Environments should implement:

```python
class Env:
    def __init__(self):
        ...
    
    def step(self, action_val):
        """
        Args:
            action_val: Action values from sheet.step()["action"]
        Returns:
            obs: Next observation (np.ndarray)
            reward: Reward signal (float)
            done: Episode done? (bool)
        """
        ...
```

## Example Usage

```python
from cortical_sheet_tensor import TensorizedCorticalSheet
from cortical_column import ColumnConfig
import numpy as np

cfg = ColumnConfig(d_in=4, d_h=47, lr_base=0.001)
sheet = TensorizedCorticalSheet(n_cols=50, cfg=cfg)

obs = np.array([0.0, 0.0, 1.0, 1.0])  # Initial obs
reward = 0.0

for step in range(500):
    result = sheet.step(obs, reward=reward)
    action = result["action"]
    obs, reward, done = env.step(action)
    if done:
        break
```

## Multi-Environment Support
System supports multiple environments:
- **Navigation**: 1D/2D goal-reaching (`bench_nav.py`, `bench_game.py`)
- **Chat**: Dialogue/QA (`bench_chat.py` - stub)
- **Tool use**: Tool selection (`bench_tool.py` - stub)

## Key Features
- Local learning (no global backprop)
- Sparse routing (top-k columns active)
- Multi-timescale plasticity (fast traces + slow consolidation)
- Engram memory layer (fast associative recall)
- Neuromodulation (DA/NE-driven learning + exploration)
