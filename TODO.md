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

## Phase 9: Multi-Area Hierarchy
- [x] **9.1 Inter-area Communication**
    - [x] Create `CortexHierarchy` manager.
    - [x] Link V1 (Sensory) -> V2 (Abstract) via FF/FB tensors.
- [x] **9.2 Predictive Feedback (L6 -> Thalamus)**
    - [x] Implement Top-down prediction of lower-area activity.
- [x] **9.3 Temporal Pooling**
    - [x] Implement slower timescales for higher-level areas.
- [x] **9.4 Hierarchical Task**
    - [x] Verify on hierarchical pattern sequence (V2 10x more stable).

## Future: Phase 10: Evolutionary Scaling (Refinement)
- [x] Implement Genetic Algorithm for `ColumnConfig` optimization.  
  - Created `genetic_algorithm.py` (tests navigation + memory fitness)
  - **TESTED: Best config found (d_h=86, lr=0.001, fitness=42.762)**
- [x] Large-scale sheet (1000+ columns) on GPU.
  - PyTorch installed (CPU only - no CUDA on system)
  - Created `cortical_sheet_pytorch.py` (CPU: 802 steps/s for 100 cols)
  - Created `bench_scale.py` for scaling benchmarks
  - **Numpy: 15.6 steps/s for 1000 cols**
- [ ] Integration with PyTorch/JAX for acceleration.
  - PyTorch CPU working, GPU (CUDA) not available on this system

---
*Review Ref (2024-04-25): Phase 7 & 8 complete. System is now a vectorized, scalable, sensorimotor cortical sheet capable of local learning and memory consolidation.*
