"""
Exp C: Scaling.
Vary swarm size (50, 100, 200 columns). Measure specialization vs size.
"""
import csv
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_swarm_exp import run_experiment
from cortical_sheet_tensor import TensorizedCorticalSheet
from cortical_column import ColumnConfig, NeuromodState


def run_exp_c(steps=1000, switch_interval=100, output_csv="exp_c_results.csv"):
    """Run Exp C: scaling swarm size, log results."""
    results = []
    swarm_sizes = [50, 100, 200]

    for n_cols in swarm_sizes:
        print(f"Running multi-task experiment with {n_cols} columns...")
        # Run multi-task experiment
        task_rewards = run_experiment(steps=steps, switch_interval=switch_interval, n_cols=n_cols)
        
        # Calculate average reward per task
        for task, rewards in task_rewards.items():
            if rewards:
                avg_reward = np.mean(rewards)
                results.append({
                    'swarm_size': n_cols,
                    'task': task,
                    'avg_reward': avg_reward
                })
                print(f"  {task:6s} | avg reward: {avg_reward:.3f}")

    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['swarm_size', 'task', 'avg_reward'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_csv}")
    return results


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    output = sys.argv[2] if len(sys.argv) > 2 else "exp_c_results.csv"
    run_exp_c(steps=steps, output_csv=output)
