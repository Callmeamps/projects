"""
Swarm Specialization Experiment (Exp A baseline)
Train a single cortical sheet on multi-task environment.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cortical_sheet_tensor import TensorizedCorticalSheet
from cortical_column import ColumnConfig, NeuromodState
from bench_swarm_specialization import MultiTaskEnv, TASKS

def run_experiment(steps=1000, switch_interval=100, n_cols=50):
    """Run multi-task training with a single sheet."""
    # Config with d_action=8 for unified action space
    cfg = ColumnConfig(
        d_in=11,  # 3 (task one-hot) + max task obs dim (8 for memory)
        d_h=64,
        d_ctx=8,
        d_action=8,  # unified action dimension
        d_lat=8,
        k_neighbors=14,
        lr_base=0.001,
    )
    sheet = TensorizedCorticalSheet(n_cols=n_cols, cfg=cfg)
    env = MultiTaskEnv(task_switch_interval=switch_interval)

    # Neuromod state for training
    nm = NeuromodState(da=1.0, ach=1.0, ne=0.5)

    total_rewards = {t: [] for t in TASKS}
    obs = env.reset()
    max_obs_dim = cfg.d_in  # 11

    print(f"Starting experiment: {steps} steps, switch every {switch_interval}")
    for i in range(steps):
        # Pad obs to d_in if needed
        if obs.shape[0] < max_obs_dim:
            padded = np.zeros(max_obs_dim)
            padded[:obs.shape[0]] = obs
            obs_in = padded
        else:
            obs_in = obs
        
        # Sheet step expects input of size d_in
        out = sheet.step(obs_in, neuromod_override=nm)
        action = out["action"]  # shape (d_action,) averaged across columns?
        # Actually action is from motor layer: shape (d_action,)
        # Ensure action is size 8
        if action.shape[0] != 8:
            # Pad if needed (should not happen)
            action = np.pad(action, (0, 8 - action.shape[0]))
        
        obs, reward, done = env.step(action)
        total_rewards[env.current_task].append(reward)

        if i % 200 == 0:
            print(f"Step {i:4d} | Task: {env.current_task} | Reward: {reward:.3f}")

    print("\nExperiment complete.")
    for t in TASKS:
        if total_rewards[t]:
            avg = np.mean(total_rewards[t])
            print(f"  {t:6s}: avg reward {avg:.3f} over {len(total_rewards[t])} steps")

if __name__ == "__main__":
    run_experiment(steps=500, switch_interval=100)
