"""
Tests for routing entropy metric.
Verifies entropy calculation from CSV logs.
"""
import os
import csv
import numpy as np
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swarm_metrics import compute_routing_entropy


def test_entropy_uniform_distribution():
    """Uniform routing scores → high entropy."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'task', 'column_id', 'routing_score', 'active'])
        # 10 columns, uniform scores 0.1 each for task 'nav'
        for col in range(10):
            writer.writerow([0, 'nav', col, 0.1, True])
    try:
        entropy = compute_routing_entropy(f.name)
        # Uniform entropy for 10 equal probs: -10*(0.1*ln(0.1)) ≈ 2.3026
        assert entropy['nav'] > 2.0, f"Expected high entropy, got {entropy['nav']}"
    finally:
        os.unlink(f.name)


def test_entropy_specialized():
    """Specialized columns → lower entropy than uniform."""
    # Create uniform CSV first to get baseline
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f_uniform:
        writer = csv.writer(f_uniform)
        writer.writerow(['step', 'task', 'column_id', 'routing_score', 'active'])
        for col in range(10):
            writer.writerow([0, 'nav', col, 0.1, True])
    # Create specialized CSV: 8 cols score 0.9, 2 cols score 0.1
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f_spec:
        writer = csv.writer(f_spec)
        writer.writerow(['step', 'task', 'column_id', 'routing_score', 'active'])
        for col in range(10):
            score = 0.9 if col < 8 else 0.1
            writer.writerow([0, 'nav', col, score, True])
    try:
        entropy_uniform = compute_routing_entropy(f_uniform.name)
        entropy_spec = compute_routing_entropy(f_spec.name)
        # Specialized entropy should be lower
        assert entropy_spec['nav'] < entropy_uniform['nav'], \
            f"Specialized entropy {entropy_spec['nav']} not lower than uniform {entropy_uniform['nav']}"
    finally:
        os.unlink(f_uniform.name)
        os.unlink(f_spec.name)


def test_entropy_calculation():
    """Entropy formula correct: -sum(p * log(p))."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'task', 'column_id', 'routing_score', 'active'])
        # 2 columns, scores 0.5 each for 'nav' → entropy = ln(2) ≈ 0.693
        writer.writerow([0, 'nav', 0, 0.5, True])
        writer.writerow([0, 'nav', 1, 0.5, True])
    try:
        entropy = compute_routing_entropy(f.name)
        expected = -2 * (0.5 * np.log(0.5))  # natural log
        assert abs(entropy['nav'] - expected) < 0.01, f"Expected {expected}, got {entropy['nav']}"
    finally:
        os.unlink(f.name)


if __name__ == "__main__":
    test_entropy_uniform_distribution()
    print("✓ test_entropy_uniform_distribution passed")
    test_entropy_specialized()
    print("✓ test_entropy_specialized passed")
    test_entropy_calculation()
    print("✓ test_entropy_calculation passed")
    print("\nAll entropy tests passed!")
