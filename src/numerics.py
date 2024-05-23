import numpy as np
import os
import time

from src.hamiltonian_updates import hamiltonian_update
from src.rounding import randomized_rounding
from src.hamiltonian_updates import gibbs_state


def compute_batch(load_dir, save_path, epsilon, hu_kwargs, offset=0):
    C_til_list = []
    opt_list = []
    result_list = []
    print('Loading files...')
    for filename in os.listdir(f'{load_dir}/'):
        data = np.load(f'{load_dir}/{filename}', allow_pickle=True).item()
        C_til_list.append(data['C_til'])
        opt_list.append(data['opt'])

    for i in range(len(C_til_list)):
        print(f'n={C_til_list[i].shape[0]}, compute {i + 1}/{len(C_til_list)}')
        start = time.time()
        try:
            feasible, H, t, delta_entropy, qc, logs = hamiltonian_update(
                C_til=C_til_list[i],
                gamma=opt_list[i] + offset,
                epsilon=epsilon,
                **hu_kwargs
            )
            result_list.append({'feasible': feasible, 'iters': t, 'delta_entropy': delta_entropy, 'qc': qc,
                                'logs': logs, 'time': time.time() - start})
        except Exception as E:
            with open('log.txt', 'a') as file:
                file.write(f'Exception: "{E}" encountered for {load_dir}/{os.listdir(f"{load_dir}/")[i]} '
                           f'with {epsilon = }, {offset = }, {hu_kwargs = } \n')

    context = {'epsilon': epsilon, 'hu_kwargs': hu_kwargs, 'offset': offset}
    np.save(f'{save_path}.npy', {'results': result_list, 'context': context})


def add_optimal_rounds(load_dir, save_dir, shots, percentile):
    for filename in sorted(os.listdir(load_dir)):
        # C_til, opt, n, s, rho_opt = np.load(f'{load_dir}/{filename}', allow_pickle=True).item()
        data: dict = np.load(f'{load_dir}/{filename}', allow_pickle=True).item()
        opt_round_res = randomized_rounding(
            data['C_til'], data['rho_opt'], shots, percentile=percentile
        )
        data['optimal_rounding'] = opt_round_res
        np.save(f'{save_dir}/{filename}', np.array(data, dtype='object'))


def hu_rounds(load_dir, save_path, log_eps, shots, percentile, hu_kwargs):
    epsilon = 10 ** (-log_eps)
    result_list = []

    for filename in sorted(os.listdir(load_dir)):
        data: dict = np.load(f'{load_dir}/{filename}', allow_pickle=True).item()
        hu_kwargs['s'] = data['context']['s']

        start = time.time()
        _, H, t, delta_entropy, qc, logs = hamiltonian_update(
            C_til=data['C_til'],
            gamma=data['opt'],
            epsilon=epsilon,
            **hu_kwargs
        )
        data['time'] = time.time() - start
        data['iters'] = t
        hu_rounding = randomized_rounding(data['C_til'], gibbs_state(H), shots, percentile=percentile)
        result_list.append({
            'iters': t, 'delta_entropy': delta_entropy, 'qc': qc,
            'logs': logs, 'time': time.time() - start,
            'optimal_rounding': data['optimal_rounding'], 'hu_rounding': hu_rounding
        })

    context = {'epsilon': epsilon, 'hu_kwargs': hu_kwargs, 'shots': shots, 'percentile': percentile}
    np.save(f'{save_path}.npy', {'results': result_list, 'context': context})
