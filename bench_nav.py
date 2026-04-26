
import numpy as np
from cortical_sheet_tensor import TensorizedCorticalSheet
from cortical_column import ColumnConfig

class NavEnv:
    def __init__(self):
        self.pos = 0.0
        self.goal = 1.0
        
    def step(self, action_val):
        old_dist = abs(self.goal - self.pos)
        # Action[0] moves right, Action[1] moves left
        move = action_val[0] - action_val[1]
        self.pos += move * 0.1
        self.pos = np.clip(self.pos, -1.0, 2.0)
        
        new_dist = abs(self.goal - self.pos)
        reward = 1.0 if new_dist < old_dist else -0.5
        done = new_dist < 0.1
        return np.array([self.pos, self.goal - self.pos]), reward, done

def run_nav():
    cfg = ColumnConfig(d_in=2, d_h=86, lr_base=0.001)
    sheet = TensorizedCorticalSheet(n_cols=50, cfg=cfg)
    env = NavEnv()
    
    total_reward = 0
    for i in range(500):
        obs, reward, done = env.step(sheet.step(obs if 'obs' in locals() else np.array([0.0, 1.0]), reward=reward if 'reward' in locals() else 0.0)["action"])
        total_reward += reward
        if i % 10 == 0:
            print(f"Step {i:2d} | Pos: {env.pos:5.2f} | Rew: {reward:5.1f}")
        if done:
            print(f"GOAL REACHED at step {i}")
            break
    print(f"Total reward: {total_reward:.1f}")

if __name__ == "__main__":
    run_nav()
