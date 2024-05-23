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
        "--offset_min",
        type=float,
        default=-0.1,
    )
    CLI.add_argument(
        "--offset_max",
        type=float,
        default=0.10001,
    )
    CLI.add_argument(
        "--offset_step",
        type=float,
        default=0.005,
    )
    CLI.add_argument(
        "--epsilon",
        nargs="*",
        type=float,
        default=[0.005],
    )

    args = CLI.parse_args()
    epsilon_list = args.epsilon

    offset_range = np.arange(args.offset_min, args.offset_max, args.offset_step)

    for epsilon in epsilon_list:
        print(f'epsilon = {round(epsilon, 6)}')
        os.makedirs(f'data/results/gamma/sparse/eps_{round(epsilon,6)}', exist_ok=True)
        os.makedirs(f'data/results/gamma/dense/eps_{round(epsilon,6)}', exist_ok=True)

        print('sparse:')
        for offset in offset_range:
            hu_kwargs['beta'] = 0.4
            print(f'gamma offset = {round(offset, 4)}')
            compute_batch(
                load_dir='data/cost_matrices/scaling/sparse',
                save_path=f'data/results/gamma/sparse/eps_{round(epsilon,6)}/offset_{round(offset, 4)}',
                epsilon=epsilon,
                hu_kwargs=hu_kwargs,
                offset=offset
            )

        print('dense:')
        for offset in offset_range:
            hu_kwargs['beta'] = 0.2
            print(f'gamma offset = {round(offset, 4)}')
            compute_batch(
                load_dir='data/cost_matrices/scaling/dense',
                save_path=f'data/results/gamma/dense/eps_{round(epsilon,6)}/offset_{round(offset, 4)}',
                epsilon=epsilon,
                hu_kwargs=hu_kwargs,
                offset=offset
            )
