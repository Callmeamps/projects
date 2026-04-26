# Cortical Microcircuit — Project TODO

## Phase 7: Full Recurrent Cortex Tile ✅
- [x] 7.1 Tensorized Cortical Sheet
- [x] 7.2 Scale to 100+ Columns
- [x] 7.3 Structural Plasticity
- [x] 7.4 Multi-timescale Plasticity
- [x] 7.5 Generative Sleep/Replay
- [x] 7.6 Local Thalamic Scaling

## Phase 8: Sensorimotor Integration ✅
- [x] 8.1 Motor Output Layer (L5b)
- [x] 8.2 Action-Selection Strategy
- [x] 8.3 Task Benchmark

## Phase 9: Multi-Area Hierarchy ✅
- [x] 9.1 Inter-area Communication
- [x] 9.2 Predictive Feedback (L6 → Thalamus)
- [x] 9.3 Temporal Pooling
- [x] 9.4 Hierarchical Task

## Phase 10: Evolutionary Scaling (Refinement) ✅
- [x] Genetic Algorithm for `ColumnConfig` optimization
  - Best: d_h=86, lr=0.001, fitness=42.762
- [x] Large-scale sheet (1000+ cols) on GPU
  - PyTorch CPU: 802 steps/s @ 100 cols
  - Numpy: 17.9 steps/s @ 1000 cols
- [x] PyTorch port (`cortical_sheet_pytorch.py`)

## Phase 11: Next Steps ⏳
- [ ] JAX port (if needed for GPU acceleration)
- [ ] Full GA run (20 gen, 20 pop)
- [x] Apply best config to navigation task
- [ ] Multi-environment testing (chat, game, tool use)
- [ ] Swarm specialization experiments
- [ ] Documentation update (API spec v2.0)

---
*Updated: 2026-04-26 | Status: Phase 10 Complete*
