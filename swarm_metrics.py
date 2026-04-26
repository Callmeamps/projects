"""
Metrics for swarm specialization analysis.
"""
import csv
import numpy as np
from collections import defaultdict


def compute_routing_entropy(csv_path):
    """
    Compute per-task routing entropy from activity CSV.
    
    Args:
        csv_path: Path to CSV with columns step, task, column_id, routing_score, active.
        
    Returns:
        dict: {task: entropy} where entropy is -sum(p * ln(p)) for normalized routing scores.
    """
    task_scores = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row['task']
            score = float(row['routing_score'])
            task_scores[task].append(score)
    
    entropy_dict = {}
    for task, scores in task_scores.items():
        scores_arr = np.array(scores)
        # Normalize to probability distribution
        total = np.sum(scores_arr)
        if total == 0:
            entropy_dict[task] = 0.0
            continue
        probs = scores_arr / total
        # Compute entropy (natural log)
        # Avoid log(0) by filtering p > 0
        mask = probs > 0
        if not np.any(mask):
            entropy_dict[task] = 0.0
            continue
        entropy = -np.sum(probs[mask] * np.log(probs[mask]))
        entropy_dict[task] = float(entropy)
    
    return entropy_dict
