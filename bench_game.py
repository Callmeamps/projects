import numpy as np
from cortical_sheet_tensor import TensorizedCorticalSheet
from cortical_column import ColumnConfig

class GridGameEnv:
    """Simple 2D grid game: agent starts at (0,0), goal at (5,5)."""
    def __init__(self):
        self.pos = np.array([0.0, 0.0])
        self.goal = np.array([5.0, 5.0])
        self.bounds = 10.0

    def step(self, action_val):
        """Action: [up, right, down, left] one-hot style."""
        move = np.array([
            action_val[0] - action_val[2],  # up - down
            action_val[1] - action_val[3]   # right - left
        ]) * 0.5
        self.pos = np.clip(self.pos + move, -self.bounds, self.bounds)

        dist = np.linalg.norm(self.goal - self.pos)
        old_dist = np.linalg.norm(self.goal - (self.pos - move))
        reward = 1.0 if dist < old_dist else -0.5
        done = dist < 0.5
        obs = np.concatenate([self.pos, self.goal - self.pos])
        return obs, reward, done

def run_game():
    cfg = ColumnConfig(d_in=4, d_h=47, lr_base=0.001)
    sheet = TensorizedCorticalSheet(n_cols=50, cfg=cfg)
    env = GridGameEnv()

    total_reward = 0
    for i in range(500):
        obs, reward, done = env.step(
            sheet.step(
                np.concatenate([env.pos, env.goal - env.pos]) if i == 0 else obs,
                reward=total_reward if i > 0 else 0.0
            )["action"]
        )
        total_reward += reward
        if i % 50 == 0:
            print(f"Step {i:3d} | Pos: {env.pos} | Dist: {np.linalg.norm(env.goal - env.pos):.2f} | Rew: {reward:5.1f}")
        if done:
            print(f"GOAL REACHED at step {i}")
            break
    print(f"Total reward: {total_reward:.1f}")

if __name__ == "__main__":
    run_game()
