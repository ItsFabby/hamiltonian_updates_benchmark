from src.cost_matrices import sample_fixed_dim
import os
import argparse

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--number",
        nargs=1,
        type=int,
        default=20,
    )
    CLI.add_argument(
        "--n",
        nargs=1,
        type=int,
        default=100,
    )
    CLI.add_argument(
        "--s",
        nargs=1,
        type=int,
        default=20,
    )
    args = CLI.parse_args()
    number = args.number
    n = args.n
    s = args.s

    os.makedirs('data/cost_matrices/comparison', exist_ok=True)
    print(f'Generating {number} cost {"matrices" if number>1 else "matrix"} for comparison.')

    sample_fixed_dim(
        n=n,
        s=s,
        number=number,
        save_dir='data/cost_matrices/comparison',
        solver_eps=10**-8,
        solver_verbose=False
    )
