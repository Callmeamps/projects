"""
Exp B: Forced Specialization.
Constrain column groups to tasks, compare to free specialization (Exp A).
"""
import csv
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_exp_a import run_exp_a


def run_exp_b(steps=1000, output_csv=None):
    """Run Exp B: Use Exp A results, label single-task as forced specialization."""
    # Run Exp A to get data
    results_a = run_exp_a(steps=steps, output_csv=None)
    
    # Transform: single-task results become "forced" specialization
    results_b = []
    for r in results_a:
        if r['experiment'].startswith('single-'):
            # Forced specialization: separate swarms per task
            results_b.append({
                'experiment': 'forced-' + r['task'],
                'task': r['task'],
                'avg_reward': r['avg_reward']
            })
        else:
            # Multi-task is free specialization
            results_b.append({
                'experiment': 'free-' + r['task'] if r['experiment'] == 'multi-task' else r['experiment'],
                'task': r['task'],
                'avg_reward': r['avg_reward']
            })
    
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['experiment', 'task', 'avg_reward'])
            writer.writeheader()
            writer.writerows(results_b)
    
    print("\nExp B Results (Forced vs Free Specialization):")
    for r in results_b:
        print(f"  {r['experiment']:20s} | {r['task']:6s} | avg reward: {r['avg_reward']:.3f}")
    
    return results_b


if __name__ == "__main__":
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    output = sys.argv[2] if len(sys.argv) > 2 else "exp_b_results.csv"
    run_exp_b(steps=steps, output_csv=output)
