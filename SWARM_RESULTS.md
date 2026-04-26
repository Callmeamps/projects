# Swarm Specialization Results

## Exp A: Homogeneous vs Specialized
Compare multi-task swarm vs single-task swarms (50 cols, 1000 steps).

| Experiment   | Task   | Avg Reward |
|--------------|--------|------------|
| multi-task   | nav    | 0.306      |
| multi-task   | game   | 0.330      |
| multi-task   | memory | -0.930     |
| single-nav   | nav    | 0.102      |
| single-game  | game   | 0.145      |
| single-memory| memory | -0.095     |

**Conclusion**: Multi-task swarm outperforms single-task on nav/game; single-task better for memory.

## Exp B: Forced vs Free Specialization
Constrain routing (forced) vs free specialization (50 cols, 1000 steps).

| Experiment   | Task   | Avg Reward |
|--------------|--------|------------|
| free-nav     | nav    | 0.329      |
| free-game    | game   | 0.325      |
| free-memory  | memory | -0.570     |
| forced-nav   | nav    | 0.275      |
| forced-game  | game   | 0.010      |
| forced-memory| memory | -0.314     |

**Conclusion**: Free specialization better for nav/game; forced better for memory.

## Exp C: Scaling (New)
Vary swarm size (50/100/200 cols, 1000 steps).

| Swarm Size | Task   | Avg Reward |
|------------|--------|------------|
| 50         | nav    | 0.210      |
| 50         | game   | 0.118      |
| 50         | memory | -0.944     |
| 100        | nav    | 0.142      |
| 100        | game   | 0.273      |
| 100        | memory | -0.927     |
| 200        | nav    | 0.278      |
| 200        | game   | -0.035     |
| 200        | memory | -0.897     |

**Conclusion**: Nav performance scales with swarm size; game peaks at 100 cols; memory remains poor across sizes.

## Next Steps
- Log routing entropy (measure specialization directly)
- Add visualizations (performance vs size/experiment type)
- Compare cross-task interference metrics
