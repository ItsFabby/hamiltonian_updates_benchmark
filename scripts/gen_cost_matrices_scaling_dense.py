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
        default=1000,
    )
    args = CLI.parse_args()
    number = args.number[0]
    n = args.n[0]

    os.makedirs('data/cost_matrices/scaling/dense', exist_ok=True)
    print(f'Generating {number} cost {"matrices" if number>1 else "matrix"} for scaling.')

    sample_fixed_dim(
        n=n,
        s=None,
        number=number,
        save_dir='data/cost_matrices/scaling/dense',
        solver_eps=10**-8,
        solver_verbose=False
    )
