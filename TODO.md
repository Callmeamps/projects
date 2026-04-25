# Cortical Microcircuit - Project TODO

## Phase 7: Full Recurrent Cortex Tile
- [x] **7.1 Tensorized Cortical Sheet**
    - [x] Migrate from Object-Oriented to Tensor-based architecture.
    - [x] Vectorize forward pass and plasticity updates.
- [x] **7.2 Scale to 100+ Columns**
    - [x] Benchmark performance: 500 cols @ 55ms.
    - [x] Implement sparse adjacency matrix (CSR).
- [x] **7.3 Structural Plasticity**
    - [x] Implement pruning and probabilistic growth.
- [x] **7.4 Multi-timescale Plasticity**
    - [x] Add Slow traces for synaptic tagging and consolidation.
- [x] **7.5 Generative Sleep/Replay**
    - [x] Implement Offline replay and consolidation.
- [x] **7.6 Local Thalamic Scaling**
    - [x] Neighborhood-based routing.

## Phase 8: Sensorimotor Integration
- [x] **8.1 Motor Output Layer (L5b)**
    - [x] Implement Dopamine-modulated motor weights.
- [x] **8.2 Action-Selection Strategy**
    - [x] Implement NE-driven exploration noise.
- [x] **8.3 Task Benchmark**
    - [x] Verify learning on 1D Navigation task (SUCCESS: 36 steps).

## Phase 9: Multi-Area Hierarchy (Future)
- [ ] Connect multiple CorticalSheets via Apical/Basal streams.
- [ ] Long-range feedback (L6 -> Thalamus).
- [ ] Large-scale behavioral tasks.

---
*Review Ref (2024-04-25): Phase 7 & 8 complete. System is now a vectorized, scalable, sensorimotor cortical sheet capable of local learning and memory consolidation.*
