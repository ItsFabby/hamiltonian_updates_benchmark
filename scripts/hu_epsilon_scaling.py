from src.numerics import compute_batch

import os

if __name__ == '__main__':
    epsilon = 0.01
    hu_kwargs_nothing = {
        's': 20,
        'b': 8,
        'beta': 0,
        'constant_step_size': True,
        'P_d_type': 1
    }

    os.makedirs('data/comparison', exist_ok=True)

    compute_batch(
        load_dir='data/cost_matrices/benchmarking',
        save_path='data/results/comparison/nothing',
        epsilon=epsilon,
        hu_kwargs=hu_kwargs_nothing
    )
