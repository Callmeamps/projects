"""
Quick run of swarm specialization multi-task env.
Verifies env works with random actions.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_swarm_specialization import MultiTaskEnv, TASKS

def run_random_episode(env, steps=200):
    """Run episode with random actions appropriate for each task."""
    obs = env.reset()
    total_reward = 0.0
    task_rewards = {t: [] for t in TASKS}
    
    for i in range(steps):
        # Generate random action based on current task
        if env.current_task == 'nav':
            action = np.random.randn(2)
        elif env.current_task == 'game':
            action = np.random.randn(4)
        else:  # memory
            action = np.random.randn(8)
        
        obs, reward, done = env.step(action)
        total_reward += reward
        task_rewards[env.current_task].append(reward)
        
        if i % 50 == 0:
            print(f"Step {i:3d} | Task: {env.current_task} | Reward: {reward:.3f}")
    
    print(f"\nTotal reward: {total_reward:.2f}")
    for t in TASKS:
        if task_rewards[t]:
            avg = np.mean(task_rewards[t])
            print(f"  {t:6s}: avg reward {avg:.3f} ({len(task_rewards[t])} steps)")
    return total_reward

if __name__ == "__main__":
    print("Running swarm specialization env test...")
    env = MultiTaskEnv(task_switch_interval=50)
    run_random_episode(env, steps=150)
    print("\nEnv works! Ready for cortical sheet integration.")
