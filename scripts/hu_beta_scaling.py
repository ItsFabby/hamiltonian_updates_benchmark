from src.numerics import compute_batch

import argparse
import numpy as np
import os

if __name__ == '__main__':
    hu_kwargs = {
        'b': 8,
        'lambda_c': 10,
        'lambda_d': 10,
        'lambda_increase': 1.3,
    }

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--beta_min",
        type=float,
        default=0.,
    )
    CLI.add_argument(
        "--beta_max",
        type=float,
        default=0.70001,
    )
    CLI.add_argument(
        "--beta_step",
        type=float,
        default=0.05,
    )
    CLI.add_argument(
        "--epsilon",
        type=float,
        default=0.005,
    )

    args = CLI.parse_args()
    epsilon = args.epsilon

    beta_range = np.arange(args.beta_min, args.beta_max, args.beta_step)

    os.makedirs('data/results/beta/sparse_l2', exist_ok=True)
    os.makedirs('data/results/beta/dense_l2', exist_ok=True)
    os.makedirs('data/results/beta/sparse_l1', exist_ok=True)
    os.makedirs('data/results/beta/dense_l1', exist_ok=True)

    print('sparse l2:')
    for beta in beta_range:
        hu_kwargs['beta'] = beta
        print(f'{beta = }')
        compute_batch(
            load_dir='data/cost_matrices/scaling/sparse',
            save_path=f'data/results/beta/sparse_l2/beta_{round(beta, 4)}',
            epsilon=epsilon,
            hu_kwargs=hu_kwargs
        )

    print('dense l2:')
    for beta in beta_range:
        hu_kwargs['beta'] = beta
        print(f'{beta = }')
        compute_batch(
            load_dir='data/cost_matrices/scaling/dense',
            save_path=f'data/results/beta/dense_l2/beta_{round(beta, 4)}',
            epsilon=epsilon,
            hu_kwargs=hu_kwargs
        )

    hu_kwargs['P_d_type'] = 1
    print('sparse l1:')
    for beta in beta_range:
        hu_kwargs['beta'] = beta
        print(f'{beta = }')
        compute_batch(
            load_dir='data/cost_matrices/scaling/sparse',
            save_path=f'data/results/beta/sparse_l1/beta_{round(beta, 4)}',
            epsilon=epsilon,
            hu_kwargs=hu_kwargs
        )

    print('dense l1:')
    for beta in beta_range:
        hu_kwargs['beta'] = beta
        print(f'{beta = }')
        compute_batch(
            load_dir='data/cost_matrices/scaling/dense',
            save_path=f'data/results/beta/dense_l1/beta_{round(beta, 4)}',
            epsilon=epsilon,
            hu_kwargs=hu_kwargs
        )
