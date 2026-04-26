"""
Tests for column activity logging.
Verifies swarm_log.csv is created with correct schema.
"""
import os
import csv
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_swarm_exp import run_experiment


def test_logging_creates_csv():
    """Running experiment creates swarm_log.csv with required columns."""
    # Use temp dir to avoid polluting project
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "swarm_log.csv")
        # Run short experiment with logging
        run_experiment(
            steps=10,
            switch_interval=5,
            output_csv=csv_path,
            n_cols=5  # small swarm for speed
        )
        # Verify CSV exists
        assert os.path.exists(csv_path), f"CSV not created at {csv_path}"
        # Verify columns
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            required = ['step', 'task', 'column_id', 'routing_score', 'active']
            for col in required:
                assert col in fieldnames, f"Missing column: {col}"
        # Verify some rows exist
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0, "No rows written"
            # Check first row has expected keys
            assert rows[0]['step'] == '0'
            assert rows[0]['column_id'] == '0'


if __name__ == "__main__":
    test_logging_creates_csv()
    print("✓ test_logging_creates_csv passed")
