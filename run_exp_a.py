"""
Exp A: Homogeneous vs Specialized.
Train multi-task swarm and 3 separate swarms. Compare performance.
"""
import csv
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_swarm_exp import run_experiment as run_multitask
from cortical_sheet_tensor import TensorizedCorticalSheet
from cortical_column import ColumnConfig, NeuromodState

# Import task environments
from bench_nav import NavEnv
from bench_game import GridGameEnv
from bench_swarm_specialization import MemoryTask


def run_single_task(task, steps=1000, n_cols=50):
    """Run a single-task experiment and return average reward."""
    # Initialize environment and get initial obs
    if task == 'nav':
        env = NavEnv()
        env.pos = 0.0
        d_in = 2
        d_action = 2
        obs = np.array([env.pos, env.goal - env.pos])
    elif task == 'game':
        env = GridGameEnv()
        env.pos = np.array([0.0, 0.0])
        d_in = 4
        d_action = 4
        obs = np.concatenate([env.pos, env.goal - env.pos])
    elif task == 'memory':
        env = MemoryTask()
        env.reset()
        d_in = 8
        d_action = 8
        obs = env.seq[0]
    else:
        raise ValueError(f"Unknown task: {task}")

    cfg = ColumnConfig(
        d_in=d_in,
        d_h=64,
        d_ctx=8,
        d_action=d_action,
        d_lat=8,
        k_neighbors=14,
        lr_base=0.001,
    )
    sheet = TensorizedCorticalSheet(n_cols=n_cols, cfg=cfg)
    nm = NeuromodState(da=1.0, ach=1.0, ne=0.5)

    total_reward = 0.0
    for i in range(steps):
        # Sheet step
        out = sheet.step(obs, neuromod_override=nm)
        action = out["action"][:d_action]
        # Environment step
        if task == 'nav':
            obs, reward, done = env.step(action)
        elif task == 'game':
            obs, reward, done = env.step(action)
        else:  # memory
            obs, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward / steps


def run_exp_a(steps=1000, output_csv=None):
    """Run Exp A and save results to CSV."""
    results = []

    # 1. Multi-task swarm
    print("Running multi-task experiment...")
    multitask_rewards = run_multitask(steps=steps, switch_interval=100, n_cols=50)
    for task, rewards in multitask_rewards.items():
        if rewards:
            avg = np.mean(rewards)
            results.append({
                'experiment': 'multi-task',
                'task': task,
                'avg_reward': avg
            })

    # 2. Separate swarms
    for task in ['nav', 'game', 'memory']:
        print(f"Running single-task experiment for {task}...")
        avg = run_single_task(task, steps=steps, n_cols=50)
        results.append({
            'experiment': f'single-{task}',
            'task': task,
            'avg_reward': avg
        })

    # Save to CSV
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['experiment', 'task', 'avg_reward'])
            writer.writeheader()
            writer.writerows(results)

    # Print comparison
    print("\nResults:")
    for r in results:
        print(f"  {r['experiment']:15s} | {r['task']:6s} | avg reward: {r['avg_reward']:.3f}")

    return results


if __name__ == "__main__":
    run_exp_a(steps=500, output_csv="exp_a_results.csv")
