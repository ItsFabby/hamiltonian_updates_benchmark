from src.numerics import compute_batch

import os

if __name__ == '__main__':
    epsilon = 0.01
    hu_kwargs_sparse = {
        's': 20,
        'b': 8,
        'lambda_c': 10,
        'lambda_d': 10,
        'beta': 0.4,
        'lambda_increase': 1.3,
    }
    hu_kwargs_dense = {
        's': 20,
        'b': 8,
        'lambda_c': 10,
        'lambda_d': 10,
        'beta': 0.4,
        'lambda_increase': 1.3,
    }

    os.makedirs('data/results', exist_ok=True)

    compute_batch(
        load_dir='data/cost_matrices/benchmarking/sparse',
        save_path='data/results/benchmarking_sparse',
        epsilon=epsilon,
        hu_kwargs=hu_kwargs_sparse
    )
    compute_batch(
        load_dir='data/cost_matrices/benchmarking/dense',
        save_path='data/results/benchmarking_dense',
        epsilon=epsilon,
        hu_kwargs=hu_kwargs_dense
    )
