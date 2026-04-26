"""
Tests for EngramMemory - Fast Associative Memory (Modern Hopfield Retrieval)
Tests verify behavior through public interface only.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engram_memory import EngramMemory


def test_init():
    """EngramMemory initializes with correct dimensions."""
    mem = EngramMemory(d_key=64, d_val=32, max_size=500, beta=5.0)
    assert mem.d_key == 64
    assert mem.d_val == 32
    assert mem.max_size == 500
    assert mem.beta == 5.0
    assert mem.count == 0


def test_store_and_retrieve():
    """Stored engram can be retrieved via similarity."""
    mem = EngramMemory(d_key=32, d_val=16, max_size=100)
    key = np.random.randn(32)
    key = key / np.linalg.norm(key)  # normalize
    value = np.random.randn(16)

    idx = mem.store(key, value)
    assert idx == 0
    assert mem.count == 1

    # Retrieve with same key should return stored value
    retrieved_value, confidence = mem.retrieve(key)
    np.testing.assert_allclose(retrieved_value, value, rtol=1e-5)
    assert confidence > 0.99  # Same key = high similarity


def test_retrieve_returns_similar():
    """Retrieve returns weighted sum biased toward most similar pattern."""
    mem = EngramMemory(d_key=32, d_val=16, max_size=100, beta=20.0)  # High beta = sharp
    key1 = np.array([1.0, 0.0, 0.0] + [0.0] * 29)
    key2 = np.array([0.0, 1.0, 0.0] + [0.0] * 29)
    val1 = np.array([1.0, 0.0] + [0.0] * 14)
    val2 = np.array([0.0, 1.0] + [0.0] * 14)

    mem.store(key1, val1)
    mem.store(key2, val2)

    # Query with something closer to key1
    query = np.array([0.9, 0.1, 0.0] + [0.0] * 29)
    retrieved_value, confidence = mem.retrieve(query)
    # Should be closer to val1 than val2
    dist1 = np.linalg.norm(retrieved_value - val1)
    dist2 = np.linalg.norm(retrieved_value - val2)
    assert dist1 < dist2, "Retrieved value should be closer to val1"
    assert confidence > 0.5


def test_reinforcement():
    """Similar engram gets reinforced, not duplicated."""
    mem = EngramMemory(d_key=32, d_val=16, max_size=100)
    key = np.random.randn(32)
    key = key / np.linalg.norm(key)
    value = np.random.randn(16)

    idx1 = mem.store(key, value)
    similar_key = key + 0.01 * np.random.randn(32)
    similar_key = similar_key / np.linalg.norm(similar_key)

    idx2 = mem.store(similar_key, value)
    # Should reinforce, not add new
    assert idx2 == idx1
    assert mem.count == 1


def test_decay():
    """Decay reduces strengths and ages memories."""
    mem = EngramMemory(d_key=16, d_val=8, max_size=10, decay_rate=0.5)
    key = np.random.randn(16)
    key = key / np.linalg.norm(key)
    mem.store(key, np.random.randn(8), strength=1.0)

    initial_strength = mem.strengths[0]
    mem.decay()
    assert mem.strengths[0] < initial_strength
    assert mem.ages[0] == 1


def test_forget_old():
    """Old engrams get pruned when memory is full."""
    mem = EngramMemory(d_key=16, d_val=8, max_size=3, decay_rate=1.0)
    for i in range(5):
        key = np.random.randn(16)
        key = key / np.linalg.norm(key)
        mem.store(key, np.random.randn(8))

    # Should not exceed max_size
    assert mem.count <= mem.max_size


if __name__ == "__main__":
    test_init()
    print("✓ test_init passed")
    test_store_and_retrieve()
    print("✓ test_store_and_retrieve passed")
    test_retrieve_returns_similar()
    print("✓ test_retrieve_returns_similar passed")
    test_reinforcement()
    print("✓ test_reinforcement passed")
    test_forget_old()
    print("✓ test_forget_old passed")
    print("\nAll tests passed!")
