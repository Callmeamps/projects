"""
Test for Exp B: Forced Specialization.
Verifies experiment runs and produces results CSV.
"""
import os
import csv
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_exp_b import run_exp_b


def test_exp_b_produces_results():
    """Running Exp B produces results CSV with required columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "exp_b_results.csv")
        # Run short experiment (100 steps)
        run_exp_b(steps=100, output_csv=results_path)
        # Check CSV exists
        assert os.path.exists(results_path), "Results CSV not created"
        # Check columns
        with open(results_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            required = ['experiment', 'task', 'avg_reward']
            for col in required:
                assert col in fieldnames, f"Missing column: {col}"
        # Check rows: forced-multi + 3 separate = 4 rows minimum
        with open(results_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) >= 4, f"Expected at least 4 rows, got {len(rows)}"


if __name__ == "__main__":
    test_exp_b_produces_results()
    print("✓ test_exp_b_produces_results passed")
