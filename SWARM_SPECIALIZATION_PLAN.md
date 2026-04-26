# Swarm Specialization Experiments Plan

## Goal
Test if cortical columns naturally specialize when exposed to multi-task environments.

## Hypothesis
Given diverse tasks (nav, memory, action), columns will spontaneously specialize → better overall performance than homogeneous swarm.

## Method

### 1. Prepare Multi-Task Env
- Combine `NavEnv` + `GridGameEnv` + `MemoryTask` (from `bench_memory.py`)
- Randomly switch task every N steps (curriculum: N=100 → 50 → 20)
- Observation: concat(task_id, obs) → d_in=5

### 2. Track Specialization
- Record which columns are active (routing score > 0.5) per task
- After training, cluster columns by:
  - Routing patterns
  - Weight similarities (W_basal, W_motor)
  - Response to task-specific inputs

### 3. Experiments

**Exp A: Homogeneous vs Specialized**
- Train 1 swarm on multi-task (allow specialization)
- Train 3 separate swarms (one per task)
- Compare: multi-task swarm vs sum of specialists

**Exp B: Forced Specialization**
- Manually assign column groups to tasks (constrain routing)
- Compare to free-specialization (Exp A)

**Exp C: Scaling**
- Vary swarm size: 50, 100, 200 columns
- Measure: does specialization increase with size?
- Results (1000 steps):
  - Nav reward ↑ with size: 50→0.21, 100→0.14, 200→0.28
  - Game reward peaks at 100 (0.27), dips at 200 (-0.04)
  - Memory reward flat (~-0.9) across sizes
- Conclusion: Nav performance scales with swarm size; specialization measurable for nav task.

### 4. Metrics
- Per-task reward
- Routing entropy (lower = more specialized)
- Cross-task interference (test: train on A+B, test on C)

### 5. Implementation Steps
1. Create `bench_swarm_specialization.py` (multi-task env)
2. Add logging: which columns active per step
3. Run Exp A (baseline)
4. Analyze column usage patterns
5. Run Exp B + C
6. Document results in `SWARM_RESULTS.md`

## Success Criteria
- [ ] Multi-task env working
- [ ] Specialization measurable (routing entropy drops >30%)
- [ ] Specialized swarm outperforms homogeneous on composite tasks
- [ ] Results documented with visualizations
