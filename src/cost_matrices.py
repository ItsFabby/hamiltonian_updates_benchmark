import numpy as np
from scipy import linalg
import time

import cvxpy as cp
import uuid


def dump_to_uuid(file_prefix, data, time_):
    fileName = file_prefix + str(uuid.uuid4()) + ".npy"

    print(f"Solved in {round(time_, 3)}s. Dumping data to", fileName)
    # np.save(fileName, data)
    np.save(fileName, data)


"""generate/solve/save C_til"""


def sample_uniform_dim(n_min, n_max, s, number, save_dir, solver_eps, solver_verbose=False):
    solver_kwargs = {'solver': cp.SCS, 'verbose': solver_verbose, 'eps': solver_eps}
    for _ in range(number):
        n = np.random.randint(n_min, n_max)
        new_C_til_dump(n, s, save_dir=save_dir, solver_kwargs=solver_kwargs)


def sample_fixed_dim(n, s, number, save_dir, solver_eps, solver_verbose=False):
    solver_kwargs = {'solver': cp.SCS, 'verbose': solver_verbose, 'eps': solver_eps}
    for _ in range(number):
        new_C_til_dump(n, s, save_dir=save_dir, solver_kwargs=solver_kwargs, include_rho=True)


def new_C_til_dump(n, s, save_dir, solver_kwargs=None, include_rho=False):
    start = time.time()
    C_til = gen_C_til(n, s)
    opt, rho_opt = solve(C_til, n, solver_kwargs=solver_kwargs)
    time_ = time.time() - start
    context = {'n': n, 's': n if s is None else s, 'solver_kwargs': solver_kwargs, 'solver_time': time_}
    if include_rho:
        dump_to_uuid(f'{save_dir}/n_{n}_s_{s}_',
                     {'C_til': C_til, 'opt': opt, 'rho_opt': rho_opt, 'context': context}, time_)
    else:
        dump_to_uuid(f'{save_dir}/n_{n}_s_{s}_',
                     {'C_til': C_til, 'opt': opt, 'rho_opt': np.nan, 'context': context}, time_)
    return {'C_til': C_til, 'opt': opt, 'context': context}


def gen_C_til(n, s=None):  # lazy implementation, gets slow if s is close to n
    C = np.random.normal(size=(n, n))
    C = (1 / 2) * (C + C.T)
    C = C - np.diag(np.diag(C))

    if s is not None and s < n:
        sparsity_mask = np.zeros((n, n))
        while np.max(np.sum(sparsity_mask, axis=1)) < s:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            sparsity_mask[i, j] = 1
            sparsity_mask[j, i] = 1
        C = C * sparsity_mask

    C = C / linalg.norm(C, ord=2)

    return C


def solve(C_til, n, solver_kwargs=None):
    if solver_kwargs is None:
        solver_kwargs = dict()
    # print('using sdp solver')
    cvx_C = C_til
    b = np.ones(n) / n

    X = cp.Variable((n, n), symmetric=True)

    constraints = [X >> 0]
    constraints += [
        X[i, i] == b[i] for i in range(n)
    ]
    # prob = cp.Problem(cp.Maximize(cp.trace(cvx_C @ X)), constraints)
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(cvx_C, cp.transpose(X)))), constraints)
    return prob.solve(**solver_kwargs), X.value

# def new_C_til(n, s, file_path=None, overwrite=True, solver_kwargs=None):
#     if not file_path:
#         file_path = f'cost_matrices/matrix_{n}_{s}'
#     if os.path.isfile(file_path + '.npy'):
#         if overwrite:
#             print('overwriting file')
#         else:
#             print('file already exists')
#             return
#     start = time.time()
#     C_til = gen_C_til(n, s=s)
#     print(f'new C_til generated in {time.time() - start}s')
# 
#     start = time.time()
#     opt, rho_opt = solve(C_til, n, solver_kwargs=solver_kwargs)
#     print(f'new C_til solved in {time.time() - start}s')
# 
#     np.save(file_path, np.array((C_til, opt, rho_opt), dtype='object'))
#     return C_til, opt, rho_opt
# 
# 
# def get_C_til(n, s, file_path=None):
#     if not file_path:
#         file_path = f'cost_matrices/matrix_{n}_{s}'
#     if os.path.isfile(file_path + '.npy'):
#         C_til, opt, rho_opt = np.load(file_path + '.npy', allow_pickle=True)
#         print('file loaded')
#     else:
#         print(
#             f'No file found. Use "new_C_til({n}, {s})" to generate a new file. '
#             f'This also runs an sdp solver, so it could take some time.')
#         raise Exception(
#             f'No file found. Use "new_C_til({n}, {s})" to generate a new file. '
#             f'This also runs an sdp solver, so it could take some time.')
#     return C_til, opt, rho_opt
# 
# 
# def solve_save(load_dir, save_dir, solver_kwargs):
#     if solver_kwargs is None:
#         solver_kwargs = {'solver': cp.SCS, 'verbose': True, 'eps': 0.01}
#     for filename in os.listdir(load_dir):
#         solve_save_one(f'{load_dir}/{filename}', f'{save_dir}/{os.path.splitext(filename)[0]}.npy', solver_kwargs)
# 
# 
# def solve_save_one(load_path, save_path, solver_kwargs):
#     print(f'solving {load_path}')
#     C_til = np.genfromtxt(load_path, delimiter=',')
#     n = C_til.shape[0]
#     opt, rho_opt = solve(C_til, n, solver_kwargs=solver_kwargs)
#     np.save(save_path, np.array((C_til, opt, rho_opt), dtype='object'))
