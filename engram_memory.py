"""
Cortical Swarm — Engram Memory Layer
Spec v1.0 Section 7.2 implementation

Provides fast associative memory (Modern Hopfield Retrieval)
for patterns, skills, and episodic traces.
"""

import numpy as np
from typing import List, Optional, Tuple

class EngramMemory:
    """
    Fast Associative Memory using Modern Hopfield-style retrieval.
    Stores (key, value) pairs where:
      - key:   Sparse activity pattern or state vector (d_h)
      - value: Compressed representation or action trace (d_v)
    """

    def __init__(self, d_key: int, d_val: int, max_size: int = 1000, 
                 beta: float = 10.0, decay_rate: float = 0.001):
        self.d_key = d_key
        self.d_val = d_val
        self.max_size = max_size
        self.beta = beta  # Inverse temperature for softmax retrieval
        self.decay_rate = decay_rate

        # Buffers
        self.keys = np.zeros((max_size, d_key))
        self.values = np.zeros((max_size, d_val))
        self.strengths = np.zeros(max_size)
        self.ages = np.zeros(max_size)
        self.count = 0

    def store(self, key: np.ndarray, value: np.ndarray, strength: float = 1.0):
        """
        Store a new engram or reinforce an existing similar one.
        """
        if self.count > 0:
            # Check for high similarity matches to reinforce
            similarities = self._cosine_sim(key, self.keys[:self.count])
            best_idx = np.argmax(similarities)
            
            if similarities[best_idx] > 0.95:
                # Reinforce existing
                self.strengths[best_idx] = 0.9 * self.strengths[best_idx] + 0.1 * strength
                self.keys[best_idx] = 0.99 * self.keys[best_idx] + 0.01 * key
                self.values[best_idx] = 0.99 * self.values[best_idx] + 0.01 * value
                self.ages[best_idx] = 0 # Reset age
                return best_idx

        # Add new
        idx = self.count
        if self.count >= self.max_size:
            # Evict weakest/oldest
            idx = np.argmin(self.strengths[:self.count] / (self.ages[:self.count] + 1))
        else:
            self.count += 1

        self.keys[idx] = key
        self.values[idx] = value
        self.strengths[idx] = strength
        self.ages[idx] = 0
        return idx

    def retrieve(self, query_key: np.ndarray, k: int = 5) -> Tuple[np.ndarray, float]:
        """
        Retrieve a weighted sum of values using Modern Hopfield interaction.
        Returns (retrieved_value, confidence).
        """
        if self.count == 0:
            return np.zeros(self.d_val), 0.0

        # Calculate similarities
        sims = self._cosine_sim(query_key, self.keys[:self.count])
        
        # Softmax weighting (Modern Hopfield)
        weights = np.exp(self.beta * sims)
        weights /= (np.sum(weights) + 1e-8)
        
        # Weighted sum of values
        retrieved = weights @ self.values[:self.count]
        
        # Confidence = max similarity
        confidence = float(np.max(sims))
        
        return retrieved, confidence

    def decay(self):
        """
        Age memories and decay strengths.
        """
        if self.count == 0: return
        self.ages[:self.count] += 1
        self.strengths[:self.count] *= (1.0 - self.decay_rate)
        
        # Prune very weak memories
        mask = self.strengths[:self.count] > 0.01
        # (For simplicity in this vectorized version, we don't shift arrays on every decay)

    def _cosine_sim(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        q_norm = np.linalg.norm(query) + 1e-8
        k_norms = np.linalg.norm(keys, axis=1) + 1e-8
        return (keys @ query) / (q_norm * k_norms)

if __name__ == "__main__":
    # Quick test
    mem = EngramMemory(d_key=8, d_val=4, max_size=10)
    
    k1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    v1 = np.array([10, 0, 0, 0], dtype=float)
    
    k2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    v2 = np.array([0, 20, 0, 0], dtype=float)
    
    mem.store(k1, v1)
    mem.store(k2, v2)
    
    # Query similar to k1
    q = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=float)
    res, conf = mem.retrieve(q)
    print(f"Query near k1 -> Value: {res}, Conf: {conf:.4f}")
    
    # Query between k1 and k2
    q_mid = np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0], dtype=float)
    res, conf = mem.retrieve(q_mid)
    print(f"Query mid k1/k2 -> Value: {res}, Conf: {conf:.4f}")
