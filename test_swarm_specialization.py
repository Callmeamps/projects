"""
Tests for Swarm Specialization multi-task environment.
Verifies basic behavior.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_swarm_specialization import MultiTaskEnv, TASKS


def test_env_init():
    """MultiTaskEnv initializes with correct number of tasks."""
    env = MultiTaskEnv()
    assert len(TASKS) == 3
    assert env.current_task in TASKS
    assert env.task_switch_interval > 0


def test_reset():
    """Reset brings env back to initial state."""
    env = MultiTaskEnv()
    env.reset()
    assert env.steps == 0
    assert env.current_task in TASKS


def test_obs_shape():
    """Observation shape is correct."""
    env = MultiTaskEnv()
    obs = env.reset()
    # Should be at least len(TASKS) + minimal task obs
    assert obs.shape[0] >= len(TASKS)


if __name__ == "__main__":
    test_env_init()
    print("✓ test_env_init passed")
    test_reset()
    print("✓ test_reset passed")
    test_obs_shape()
    print("✓ test_obs_shape passed")
    print("\nAll tests passed!")
