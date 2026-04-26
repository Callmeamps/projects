"""
Test for Exp A: Homogeneous vs Specialized.
Verifies experiment runs and produces results CSV.
"""
import os
import csv
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_exp_a import run_exp_a


def test_exp_a_produces_results():
    """Running Exp A produces results CSV with required columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "exp_a_results.csv")
        # Run short experiment (100 steps)
        run_exp_a(steps=100, output_csv=results_path)
        # Check CSV exists
        assert os.path.exists(results_path), "Results CSV not created"
        # Check columns
        with open(results_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            required = ['experiment', 'task', 'avg_reward']
            for col in required:
                assert col in fieldnames, f"Missing column: {col}"
        # Check rows: multi-task + 3 separate = 4 rows minimum
        with open(results_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) >= 4, f"Expected at least 4 rows, got {len(rows)}"


if __name__ == "__main__":
    test_exp_a_produces_results()
    print("✓ test_exp_a_produces_results passed")
