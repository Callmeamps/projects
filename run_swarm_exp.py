"""
Swarm Specialization Experiment (Exp A baseline)
Train a single cortical sheet on multi-task environment.
"""
import csv
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cortical_sheet_tensor import TensorizedCorticalSheet
from cortical_column import ColumnConfig, NeuromodState
from bench_swarm_specialization import MultiTaskEnv, TASKS


def run_experiment(steps=1000, switch_interval=100, n_cols=50, output_csv=None):
    """Run multi-task training with a single sheet.
    
    If output_csv is provided, logs per-column activity to CSV.
    """
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

    # Open CSV if requested
    csv_file = None
    csv_writer = None
    if output_csv:
        csv_file = open(output_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'task', 'column_id', 'routing_score', 'active'])

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
        action = out["action"]  # shape (d_action,)
        # Ensure action is size 8
        if action.shape[0] != 8:
            action = np.pad(action, (0, 8 - action.shape[0]))

        obs, reward, done = env.step(action)
        total_rewards[env.current_task].append(reward)

        # Log to CSV if enabled
        if csv_writer:
            for col_idx in range(n_cols):
                csv_writer.writerow([
                    i,
                    env.current_task,
                    col_idx,
                    float(sheet.route_scores[col_idx]),
                    bool(sheet.active_mask[col_idx])
                ])

        if i % 200 == 0:
            print(f"Step {i:4d} | Task: {env.current_task} | Reward: {reward:.3f}")

    # Close CSV
    if csv_file:
        csv_file.close()

    print("\nExperiment complete.")
    for t in TASKS:
        if total_rewards[t]:
            avg = np.mean(total_rewards[t])
            print(f"  {t:6s}: avg reward {avg:.3f} over {len(total_rewards[t])} steps")

    return total_rewards


if __name__ == "__main__":
    run_experiment(steps=500, switch_interval=100)
