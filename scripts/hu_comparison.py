from src.numerics import compute_batch
from src.hu_binary_search import bin_search_batch

import argparse
import os

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
    )
    CLI.add_argument(
        "--beta",
        type=float,
        default=0.45,
    )
    args = CLI.parse_args()
    epsilon = args.epsilon
    beta = args.beta

    hu_kwargs_constant = {
        'b': 8,
        'beta': 0,
        'constant_step_size': True,
        'P_d_type': 1
    }
    hu_kwargs_adaptive = {
        'b': 8,
        'beta': 0,
        'P_d_type': 1
    }
    hu_kwargs_P_d_2 = {
        'b': 8,
        'beta': 0,
        'P_d_type': 2
    }
    hu_kwargs_momentum = {
        'b': 8,
        'beta': beta,
        'P_d_type': 2
    }

    os.makedirs(f'data/results/comparison', exist_ok=True)

    compute_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/constant_feasible',
        epsilon=epsilon,
        hu_kwargs=hu_kwargs_constant,
        offset=0
    )
    compute_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/adaptive_feasible',
        epsilon=epsilon,
        hu_kwargs=hu_kwargs_adaptive,
        offset=0
    )
    compute_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/P_d_2_feasible',
        epsilon=epsilon,
        hu_kwargs=hu_kwargs_P_d_2,
        offset=0
    )
    compute_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/momentum_feasible',
        epsilon=epsilon,
        hu_kwargs=hu_kwargs_momentum,
        offset=0
    )

    compute_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/constant_infeasible',
        epsilon=epsilon,
        hu_kwargs=hu_kwargs_constant,
        offset=0.02
    )
    compute_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/adaptive_infeasible',
        epsilon=epsilon,
        hu_kwargs=hu_kwargs_adaptive,
        offset=0.02
    )
    compute_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/P_d_2_infeasible',
        epsilon=epsilon,
        hu_kwargs=hu_kwargs_P_d_2,
        offset=0.02
    )
    compute_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/momentum_infeasible',
        epsilon=epsilon,
        hu_kwargs=hu_kwargs_momentum,
        offset=0.02
    )

    bin_search_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/adaptive_bin_search',
        epsilon=epsilon,
        gap=epsilon,
        hu_kwargs=hu_kwargs_adaptive
    )
    bin_search_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/constant_bin_search',
        epsilon=epsilon,
        gap=epsilon,
        hu_kwargs=hu_kwargs_constant
    )
    bin_search_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/P_d_2_bin_search',
        epsilon=epsilon,
        gap=epsilon,
        hu_kwargs=hu_kwargs_P_d_2
    )
    bin_search_batch(
        load_dir='data/cost_matrices/comparison',
        save_path=f'data/results/comparison/momentum_bin_search',
        epsilon=epsilon,
        gap=epsilon,
        hu_kwargs=hu_kwargs_momentum
    )
