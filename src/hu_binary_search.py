import numpy as np
import time
import os

from src.hamiltonian_updates import hamiltonian_update


def binary_search_HU(C_til, epsilon, gap, weight_ratio=1, gamma_min=-1, gamma_max=1, hu_kwargs=None):
    if hu_kwargs is None:
        hu_kwargs = dict()

    bin_counter = 0
    w_min = weight_ratio / (weight_ratio + 1)
    w_max = 1 - w_min

    bin_accepted = list()
    bin_declined = list()
    s = np.max(np.count_nonzero(C_til, axis=1))

    while gamma_max - gamma_min > gap:
        gamma = w_max * gamma_max + w_min * gamma_min
        time_start = time.time()
        is_accepted, H, iters, delta_entropy, qc, logs = hamiltonian_update(
            C_til, gamma, epsilon, s=s, **hu_kwargs
        )
        if is_accepted:
            bin_accepted.append({'feasible': True, 'iters': iters, 'delta_entropy': delta_entropy, 'qc': qc,
                                 'logs': logs, 'time': time.time() - time_start})
            gamma_min = gamma

        else:
            bin_declined.append({'feasible': False, 'iters': iters, 'delta_entropy': delta_entropy, 'qc': qc,
                                 'logs': logs, 'time': time.time() - time_start})
            gamma_max = gamma
        bin_counter += 1
    print(f'{gamma_min = }, {gamma_max = }')
    return gamma_min, bin_counter, bin_accepted, bin_declined


def bin_search_batch(load_dir, save_path, epsilon, gap, hu_kwargs=None, bin_search_kwargs=None):
    if bin_search_kwargs is None:
        bin_search_kwargs = dict()

    C_til_list = []
    result_list = []

    for filename in os.listdir(f'{load_dir}/'):
        data: dict = np.load(f'{load_dir}/{filename}', allow_pickle=True).item()
        C_til_list.append(data['C_til'])

    for i in range(len(C_til_list)):
        print(f'n={C_til_list[i].shape[0]}, compute {i + 1}/{len(C_til_list)}')
        time_start = time.time()
        # try:
        gamma_min, _, bin_accepted, bin_declined = binary_search_HU(
            C_til=C_til_list[i],
            epsilon=epsilon,
            gap=gap,
            hu_kwargs=hu_kwargs,
            **bin_search_kwargs
        )
        result_list.append({'gamma_min': gamma_min, 'bin_accepted': bin_accepted, 'bin_declined': bin_declined,
                            'total_time': time_start - time.time()})
        # except Exception as E:
        #     with open('log', 'a') as file:
        #         file.write(
        #             f'Exception: "{E}" encountered in binary search for {load_dir}/{os.listdir(f"{load_dir}/")[i]}'
        #             f'with {epsilon = }, {hu_kwargs = } \n')

    context = {'epsilon': epsilon, 'gap': gap, 'hu_kwargs': hu_kwargs, 'bin_search_kwargs': bin_search_kwargs}
    np.save(f'{save_path}.npy', {'results': result_list, 'context': context})