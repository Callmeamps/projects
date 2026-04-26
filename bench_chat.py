"""Simple chat environment stub for multi-environment testing."""
import numpy as np
from cortical_sheet_tensor import TensorizedCorticalSheet
from cortical_column import ColumnConfig

class ChatEnv:
    """Simple chat: agent responds to queries with yes/no/unknown."""
    def __init__(self):
        self.queries = [
            ("Is sky blue?", "yes"),
            ("Is grass green?", "yes"),
            ("Is fire cold?", "no"),
        ]
        self.idx = 0

    def step(self, action_val):
        if self.idx >= len(self.queries):
            return None, 0.0, True  # Done

        query, expected = self.queries[self.idx]
        # Action: [yes, no, unknown]
        resp = "yes" if action_val[0] > action_val[1] else "no" if action_val[1] > action_val[2] else "unknown"
        reward = 1.0 if resp == expected else -0.5
        done = self.idx >= len(self.queries) - 1
        self.idx += 1
        obs = np.array([self.idx / len(self.queries)] + [0.0, 0.0, 0.0])[:4]  # Simple obs
        return obs, reward, done

def run_chat():
    cfg = ColumnConfig(d_in=4, d_h=47, lr_base=0.001)
    sheet = TensorizedCorticalSheet(n_cols=50, cfg=cfg)
    env = ChatEnv()
    # TODO: Implement full chat loop
    print("Chat environment stub created. TODO: implement full loop.")

if __name__ == "__main__":
    run_chat()
