import numpy as np

from src.gate_count import QuantumComputer

gpu = False
try:
    import cupy
    from cupyx.scipy import linalg

    gpu = True
    print('Using GPU with Cuda')
except ImportError:
    from scipy import linalg

    print('Cupy not installed. Using CPU instead.')

if gpu:
    def gibbs_state(H):
        H_cp = cupy.asarray(H)
        exp_ = cupy.asnumpy(linalg.expm(-H_cp))
        # if np.random.randint(10000) == 0: print(np.trace(exp_))
        return exp_ / np.trace(exp_)
else:
    def gibbs_state(H):
        exp_ = linalg.expm(-H)
        # if np.random.randint(1000) == 1: print(np.trace(exp_))
        return exp_ / np.trace(exp_)


def centralize_H(H):
    n = np.shape(H)[0]
    return H - np.trace(H) / n * np.eye(n)


def T_max(n, error_parameter):
    return int(64 * np.ceil(np.log2(n) * (1 / error_parameter ** 2)) + 1)


def tr_prod(A, B):
    return np.sum(A * B.T)


def get_P_d(diag_dev, n, P_d_type):
    if P_d_type == 2:
        norm_ = np.max(np.abs(diag_dev))
        if norm_ == 0:
            return np.zeros((n, n))
        return np.diag(diag_dev) / norm_
    if P_d_type == 1:
        return np.sign(np.diag(diag_dev)) - np.trace(np.sign(np.diag(diag_dev))) / n * np.eye(n)
    raise Exception(f'Invalid P_d type: {P_d_type}. Valid types are 1 and 2.')


def dist_c(rho, P_c):
    return tr_prod(P_c, rho)


# def dist_d(diag_dev):
#     return np.sum(diag_dev ** 2)


def diag_deviations(rho, n):
    return np.diag(rho) - np.ones(n) / n


"""Hamiltonian updates"""


def hamiltonian_update(
        C_til, gamma, epsilon, b=None, s=None,
        lambda_c=10, lambda_d=10, beta=0.4, lambda_increase=1.3, P_d_type=2, constant_step_size=False,
        print_resolution=False, save_as=None, verbose=False
):
    s = np.max(np.count_nonzero(C_til, axis=1))
    n = C_til.shape[0]

    P_c = - C_til + gamma * np.eye(n)  # constant 

    qc = QuantumComputer(b, n, s)

    H = np.zeros(shape=(n, n))
    rho = np.eye(n) / n
    momentum = np.zeros(shape=(n, n))

    # T is the original theoretical upper bound for the number of iterations. Note, for our improvements this is not
    # necessarily true, but one should still never exceed it.
    T = T_max(n, epsilon)
    delta_entropy = 0
    max_entropy = np.log2(n)

    distance_c = tr_prod(P_c, np.eye(n) / n)
    l1_d = 0

    logs = init_logs()

    for t in range(T):
        if print_resolution:
            print_progress(print_resolution, verbose, t, distance_c, l1_d, delta_entropy, lambda_c, lambda_d, logs)

        if delta_entropy > max_entropy:
            # entropy estimation is higher than log2(n) -> there is no feasible solution
            print_results(verbose, False, gamma, t, delta_entropy, logs)
            return False, H, t, delta_entropy, qc, logs

        qc.sample_trArho(H, epsilon)
        distance_c = tr_prod(P_c, rho)

        if distance_c >= epsilon:
            # c update
            if constant_step_size:
                Delta_H = P_c + beta / lambda_c * momentum
            else:
                Delta_H = distance_c * P_c + beta / lambda_c * momentum
            H, rho, delta_entropy, lambda_c, logs = update(
                qc, H, Delta_H, lambda_c, delta_entropy, epsilon, lambda_increase, logs, 'c', constant_step_size
            )
            momentum = lambda_c * Delta_H
            continue

        qc.sample_diag_dev(H, epsilon)
        diag_dev = diag_deviations(rho, n)

        l1_d = np.sum(np.abs(diag_dev))
        if l1_d >= epsilon:
            # d update
            P_d = get_P_d(diag_dev, n, P_d_type)
            if distance_c < 0:
                Delta_H = P_d
            else:
                Delta_H = P_d + beta / lambda_d * momentum

            H, rho, delta_entropy, lambda_d, logs = update(
                qc, H, Delta_H, lambda_d, delta_entropy, epsilon, lambda_increase, logs, 'd', constant_step_size
            )
            momentum = lambda_d * Delta_H
            continue

        # We know both distances are below the error tolerance -> HU found a feasible solution
        if save_as is not None:
            np.save(f'results_HU/{save_as}_{gamma}_{epsilon}',
                    np.array((H, C_til, delta_entropy, t), dtype='object'))

        print_results(verbose, True, gamma, t, delta_entropy, logs)
        return True, H, t, delta_entropy, qc, logs

    raise Exception('HU reached maximum number of iterations.')


def update(qc, H, Delta_H, lambda_, delta_entropy, epsilon, lambda_increase, logs, update_type, constant_step_size):
    if constant_step_size:
        lambda_ = epsilon / 16
    H_new = centralize_H(H + lambda_ * Delta_H)
    rho_new = gibbs_state(H_new)

    qc.sample_trArho(H_new, epsilon)
    dist_mom_new = tr_prod(rho_new, Delta_H)

    # test for overshoots
    while dist_mom_new < 0 and not constant_step_size:
        logs[f'overshot_{update_type}'] += 1
        lambda_ *= 0.5

        H_new = H + lambda_ * Delta_H
        rho_new = gibbs_state(H_new)

        qc.sample_trArho(H_new, epsilon)
        dist_mom_new = tr_prod(rho_new, Delta_H)

    # estimate the change in entropy
    rho_half = gibbs_state(H + lambda_ * Delta_H / 2)
    qc.sample_trArho(H + lambda_ * Delta_H / 2, epsilon)

    entropy_change = lambda_ / 2 * (tr_prod(rho_half, Delta_H) + dist_mom_new)
    delta_entropy += entropy_change

    # logs and sanity checks
    if entropy_change <= dist_mom_new ** 2 / 4:
        logs['entro_counter'] += 1
    logs['c_steps'].append(lambda_)
    if np.isnan(dist_mom_new):
        raise Exception(f'NaN value after update')

    # increasing the step factor for the next iteration
    lambda_ *= lambda_increase

    return H_new, rho_new, delta_entropy, lambda_, logs


def init_logs():
    logs = dict()
    logs['c_steps'] = list()
    logs['d_steps'] = list()
    logs['d_overshoots'] = list()
    logs['overshot_c'] = 0
    logs['overshot_d'] = 0
    logs['entro_counter'] = 0  # if this is not 0 at the end, it's a strong indicator that something went wrong
    return logs


def print_progress(print_resolution, verbose, t, distance_c, l1_d, delta_entropy, lambda_c, lambda_d, logs):
    if t % print_resolution == 0 and t > 0:
        if verbose:
            print(f't: {t},', f'distance_c: {round(distance_c, 6)}', f'l1_distance_d: {round(l1_d, 6)}',
                  f'delta_entropy: {round(delta_entropy, 4)}',
                  f'lambda_c: {round(lambda_c, 4)}', f'lambda_d: {round(lambda_d, 4)},',
                  f'overshot_c: {logs["overshot_c"]}',
                  f'overshot_d: {logs["overshot_d"]},',
                  f'entropy_counter: {logs["entro_counter"]}', sep='    \t')
        else:
            print(f't: {t},', f'distance_c: {round(distance_c, 6)}', f'l1_distance_d: {round(l1_d, 6)}',
                  f'delta_entropy: {round(delta_entropy, 4)}')


def print_results(verbose, is_success, gamma, t, delta_entropy, logs):
    if verbose:
        print(f'{"success!" if is_success else "infeasible!"}', f'gamma: {round(gamma, 6)}'.ljust(18, ' '), f't: {t},',
              f'overshot_c: {logs["overshot_c"]}', f'overshot_d: {logs["overshot_d"]},',
              f'delta_entropy: {round(delta_entropy, 4)}', f'entropy_counter: {logs["entro_counter"]}', sep='    \t')
    else:
        print(f'{"success!" if is_success else "infeasible!"}',
              f'gamma: {round(gamma, 6)}'.ljust(18, ' '), f't: {t},', f'delta_entropy: {round(delta_entropy, 4)}',
              sep='    \t')
