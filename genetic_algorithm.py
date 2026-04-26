"""
Phase 10.1: Genetic Algorithm for ColumnConfig Optimization

Optimizes cortical column hyperparameters using evolutionary search.
Fitness based on navigation task performance + memory benchmark.
"""

import numpy as np
from typing import List, Tuple
import copy

from cortical_column import ColumnConfig
from cortical_sheet_tensor import TensorizedCorticalSheet
from bench_nav import run_nav, NavEnv
from bench_memory import run_memory_benchmark


class Individual:
    """Represents a candidate ColumnConfig with fitness score."""
    def __init__(self, cfg: ColumnConfig, fitness: float = -float('inf')):
        self.cfg = cfg
        self.fitness = fitness
        self.genes = self._config_to_genes()

    def _config_to_genes(self) -> np.ndarray:
        """Convert config to normalized gene vector [0,1]."""
        return np.array([
            self.cfg.d_h / 128,           # 16-128
            self.cfg.lr_base * 1000,       # 0.001-0.1
            self.cfg.tau_elig / 100,       # 5-100
            self.cfg.anti_hebb_scale * 20,  # 0-0.5
            self.cfg.target_firing_rate * 100,  # 0.01-0.1
            self.cfg.k_neighbors / 16,      # 2-16
        ])

    @staticmethod
    def from_genes(genes: np.ndarray) -> 'Individual':
        """Create Individual from gene vector."""
        cfg = ColumnConfig(
            d_in=2,  # Nav task uses 2D input
            d_h=max(16, min(128, int(genes[0] * 128))),
            d_ctx=8,
            d_lat=8,
            k_neighbors=max(2, min(16, int(genes[5] * 16))),
            lr_base=max(0.001, min(0.1, genes[1] / 1000)),
            tau_elig=max(5, min(100, int(genes[2] * 100))),
            anti_hebb_scale=max(0.0, min(0.5, genes[3] / 20)),
            target_firing_rate=max(0.01, min(0.1, genes[4] / 100)),
        )
        return Individual(cfg)


def evaluate_fitness(cfg: ColumnConfig) -> float:
    """
    Fitness = nav performance (reward) + memory stability (error reduction).
    Higher = better.
    """
    fitness = 0.0

    # Task 1: Navigation (weight: 0.7)
    try:
        env = NavEnv()
        sheet = TensorizedCorticalSheet(n_cols=30, cfg=cfg)  # Smaller for speed
        total_reward = 0.0
        steps_to_goal = 100
        obs = np.array([0.0, 1.0])

        for i in range(100):
            reward = 0.0 if i == 0 else total_reward
            out = sheet.step(obs, reward=reward)
            obs, r, done = env.step(out["action"])
            total_reward += r
            if done and steps_to_goal == 100:
                steps_to_goal = i
                break

        # Fitness: reward (maximize) + penalty for slow convergence
        nav_score = total_reward + (100 - steps_to_goal) * 0.1
        fitness += 0.7 * nav_score
    except Exception as e:
        print(f"Nav eval failed: {e}")
        nav_score = -100

    # Task 2: Memory (weight: 0.3)
    try:
        from cortical_column import NeuromodState
        sheet = TensorizedCorticalSheet(n_cols=30, cfg=cfg)
        rng = np.random.default_rng(42)
        pattern = rng.normal(0, 1, cfg.d_in)
        errors = []

        nm = NeuromodState(da=1.5, ach=1.2)
        for i in range(50):
            out = sheet.step(pattern, neuromod_override=nm)
            errors.append(out["mean_e_mag"])

        mem_score = -np.mean(errors)  # Lower error = higher fitness
        fitness += 0.3 * mem_score
    except Exception as e:
        print(f"Memory eval failed: {e}")
        mem_score = -10

    return fitness


def mutate(individual: Individual, mutation_rate: float = 0.3,
           sigma: float = 0.1) -> Individual:
    """Apply Gaussian mutation to genes."""
    genes = individual.genes.copy()
    for i in range(len(genes)):
        if np.random.random() < mutation_rate:
            genes[i] = np.clip(genes[i] + np.random.normal(0, sigma), 0.0, 1.0)
    return Individual.from_genes(genes)


def crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """Uniform crossover."""
    genes1 = parent1.genes.copy()
    genes2 = parent2.genes.copy()

    for i in range(len(genes1)):
        if np.random.random() < 0.5:
            genes1[i], genes2[i] = genes2[i], genes1[i]

    return Individual.from_genes(genes1), Individual.from_genes(genes2)


def tournament_select(population: List[Individual], k: int = 3) -> Individual:
    """Tournament selection."""
    candidates = np.random.choice(population, k, replace=False)
    return max(candidates, key=lambda ind: ind.fitness)


def run_evolution(generations: int = 20, pop_size: int = 20,
                  elite_size: int = 2) -> Individual:
    """Run genetic algorithm."""

    # Initialize population
    population = []
    for _ in range(pop_size):
        genes = np.random.random(6)  # Random genes in [0,1]
        ind = Individual.from_genes(genes)
        population.append(ind)

    # Evaluate initial population
    print("Evaluating initial population...")
    for ind in population:
        ind.fitness = evaluate_fitness(ind.cfg)
        print(f"  Fitness: {ind.fitness:.3f} | Config: d_h={ind.cfg.d_h}, lr={ind.cfg.lr_base:.4f}")

    best_overall = max(population, key=lambda ind: ind.fitness)

    # Evolution loop
    for gen in range(generations):
        print(f"\n=== Generation {gen + 1}/{generations} ===")

        # Sort by fitness
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        # Elitism: keep best individuals
        new_population = population[:elite_size]

        # Generate offspring
        while len(new_population) < pop_size:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1)
            child2 = mutate(child2)

            child1.fitness = evaluate_fitness(child1.cfg)
            child2.fitness = evaluate_fitness(child2.cfg)

            new_population.extend([child1, child2])

        population = new_population[:pop_size]

        # Track best
        gen_best = max(population, key=lambda ind: ind.fitness)
        if gen_best.fitness > best_overall.fitness:
            best_overall = gen_best

        print(f"  Best fitness: {gen_best.fitness:.3f}")
        print(f"  Best config: d_h={gen_best.cfg.d_h}, lr={gen_best.cfg.lr_base:.4f}")

    print(f"\n=== Evolution Complete ===")
    print(f"Best overall fitness: {best_overall.fitness:.3f}")
    print(f"Best config: {best_overall.cfg}")

    return best_overall


if __name__ == "__main__":
    print("Starting Genetic Algorithm for ColumnConfig Optimization...")
    best = run_evolution(generations=20, pop_size=20, elite_size=2)

    # Save best config
    import json
    config_dict = {
        'd_in': best.cfg.d_in,
        'd_h': best.cfg.d_h,
        'd_ctx': best.cfg.d_ctx,
        'd_lat': best.cfg.d_lat,
        'k_neighbors': best.cfg.k_neighbors,
        'lr_base': best.cfg.lr_base,
        'tau_elig': best.cfg.tau_elig,
        'anti_hebb_scale': best.cfg.anti_hebb_scale,
        'target_firing_rate': best.cfg.target_firing_rate,
    }

    with open('best_column_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nSaved best config to best_column_config.json")
