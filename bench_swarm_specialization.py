"""
Swarm Specialization Experiments
Multi-task environment combining navigation, game, and memory tasks.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_nav import NavEnv
from bench_game import GridGameEnv

# Task identifiers
TASKS = ['nav', 'game', 'memory']


class MemoryTask:
    """Simple associative memory task: predict next pattern in sequence."""
    def __init__(self):
        rng = np.random.default_rng(123)
        self.patterns = [rng.normal(0, 1, 8) for _ in range(3)]
        self.seq = (self.patterns * 34)  # 102 steps
        self.step_idx = 0
        self.pos = 0.0  # dummy for compatibility

    def reset(self):
        self.step_idx = 0
        self.pos = 0.0

    def step(self, action=None):
        """Return current pattern as obs. Reward based on prediction accuracy if action given."""
        obs = self.seq[self.step_idx % len(self.seq)]
        if action is not None:
            # Action is predicted next pattern; reward = negative MSE
            next_pat = self.seq[(self.step_idx + 1) % len(self.seq)]
            reward = -np.mean((action - next_pat) ** 2)
        else:
            reward = 0.0
        self.step_idx += 1
        done = self.step_idx >= len(self.seq)
        return obs, reward, done


class MultiTaskEnv:
    """
    Multi-task environment that switches between nav, game, and memory tasks.
    Observation: concat(task_one_hot, task_observation)
    Action: expects a unified vector of size 8.
            First 2 dims used for nav, first 4 for game, all 8 for memory.
    """
    def __init__(self, task_switch_interval=100):
        self.task_switch_interval = task_switch_interval
        self.steps = 0
        self.envs = {
            'nav': NavEnv(),
            'game': GridGameEnv(),
            'memory': MemoryTask(),
        }
        self.current_task = np.random.choice(TASKS)
        self._reset_current()

    def _reset_current(self):
        if self.current_task == 'nav':
            self.envs['nav'].pos = 0.0
        elif self.current_task == 'game':
            self.envs['game'].pos = np.array([0.0, 0.0])
        else:
            self.envs['memory'].reset()

    def reset(self):
        self.steps = 0
        self.current_task = np.random.choice(TASKS)
        self._reset_current()
        return self._get_obs()

    def step(self, action):
        """
        action: unified vector of size 8.
        """
        self.steps += 1
        # Switch task if interval reached
        if self.steps % self.task_switch_interval == 0:
            self.current_task = np.random.choice(TASKS)
            self._reset_current()

        env = self.envs[self.current_task]
        # Slice action according to task
        if self.current_task == 'nav':
            task_action = action[:2]
        elif self.current_task == 'game':
            task_action = action[:4]
        else:  # memory
            task_action = action[:8]

        if self.current_task == 'nav':
            obs, reward, done = env.step(task_action)
        elif self.current_task == 'game':
            obs, reward, done = env.step(task_action)
        else:  # memory
            obs, reward, done = env.step(task_action)

        return self._augment_obs(obs), reward, done

    def _augment_obs(self, task_obs):
        """Concatenate task one-hot with task observation."""
        task_one_hot = np.array([1.0 if t == self.current_task else 0.0 for t in TASKS])
        # Ensure task_obs is flat
        task_obs = np.atleast_1d(task_obs).flatten()
        return np.concatenate([task_one_hot, task_obs])

    def _get_obs(self):
        """Get observation without stepping (for reset)."""
        env = self.envs[self.current_task]
        if self.current_task == 'nav':
            task_obs = np.array([env.pos, env.goal - env.pos])
        elif self.current_task == 'game':
            task_obs = np.concatenate([env.pos, env.goal - env.pos])
        else:
            task_obs = env.seq[env.step_idx % len(env.seq)]
        return self._augment_obs(task_obs)


if __name__ == "__main__":
    # Quick sanity check
    env = MultiTaskEnv(task_switch_interval=50)
    print("Initial task:", env.current_task)
    # Use unified action vector of size 8
    dummy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    obs, reward, done = env.step(dummy_action)
    print("Obs shape:", obs.shape)
    print("Task one-hot:", obs[:3])
    print("Success: MultiTaskEnv ready.")
