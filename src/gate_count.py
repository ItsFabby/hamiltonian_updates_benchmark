import numpy as np

CNOT = np.array([1, 0, 0, 0])
T_GATE = np.array([0, 1, 0, 0])
O_F = np.array([0, 0, 1, 0])
O_H = np.array([0, 0, 0, 1])

TOFFOLI = 6 * CNOT + 7 * T_GATE
CONTROLLED_SWAP = TOFFOLI + 2 * CNOT


def comparer_cost(b):
    return (2 * b - 1) * TOFFOLI + (4 * b - 3) * CNOT


def lem15_cost(b):
    return comparer_cost(b)


def multi_bit_toffoli_cost(m):
    return (2 * m - 1) * TOFFOLI


def cU_cost(b, n):
    return 2 * O_F + 4 * O_H + 2 * lem15_cost(b) + 2 * np.ceil(np.log2(n)) * CONTROLLED_SWAP


def cW_cost(b, n):
    return 2 * cU_cost(b, n) + multi_bit_toffoli_cost(2 * b + np.ceil(np.log2(n)) + 3) + 3 * CNOT


def cU_H_cost(b, n, t, s, H_infnorm, epsilon):
    # epsilon_prime = epsilon ** 2 # no idea why this was here
    Q = get_q(t * s * H_infnorm, epsilon) - 1
    return Q * cW_cost(b, n) + 2 * Q * CNOT + 2 * Q * TOFFOLI


def get_q(t, epsilon):
    # find min q with a binary search
    def large_enough(q, t_):
        # return epsilon >= 4*(t)**q / (2**q * np.math.factorial(int(q)))   # factorial becomes too large
        # using Stirling's approximation:
        return q * (1 + np.log2(q / (np.e * t_))) - 0.5 * np.log2(2 * np.pi * q) >= 2 + np.log2(1 / epsilon)

    q_max = np.ceil(max(np.e * t, 2 + np.log2(1 / epsilon)))
    q_min = 0

    while q_max - q_min >= 1:
        q_mid = (q_max + q_min) / 2
        if large_enough(q_mid, t):
            q_max = q_mid
        else:
            q_min = q_mid
    return np.ceil(q_min)


def cont_sim_cost(M, b, n, angle, s, H_infnorm, epsilon):
    J = int(np.ceil(np.log2(M)))
    return sum([cU_H_cost(b, n, 2 ** j * angle, s, H_infnorm, 2 ** (j - J - 1) * epsilon) for j in range(J + 1)])


def gibbs_cost(b, n, s, H_infnorm, epsilon):
    M = 2 * int(np.ceil(np.log(8 / epsilon) * (2 + H_infnorm)))
    angle = np.pi / (H_infnorm + 1)
    return 9 / 4 * np.e * np.sqrt(n) * cont_sim_cost(M, b, n, angle, s, H_infnorm, epsilon / 2)


def trArho_cost(b, n, s, H_infnorm, epsilon):
    epsilon_prime = epsilon / 19
    M = 2 * int(np.ceil(np.log(8 / epsilon_prime) * (2 + H_infnorm)))
    angle = np.pi / (H_infnorm + 1)
    return 9 / 2 * np.e * np.sqrt(n) * cont_sim_cost(M, b, n, angle, s, H_infnorm, epsilon_prime) / epsilon_prime


def H_plus(H, n):
    return 2 * H + 3 / 2 * np.eye(n)


class QuantumComputer:
    def __init__(self, b, n, s):
        self.cost = np.zeros(4)
        self.count_d = 0
        self.count_c = 0
        self.norms = []
        self.b = b
        self.n = int(n)  # we get overflow problems with int32
        self.s = s

        if self.b is None:
            print('Can\'t count gates without value given for b')
            self.b = 0
        if self.s is None:
            print('Can\'t count gates without value given for s')
            self.s = 0

    def sample_diag_dev(self, H, epsilon):
        if self.s == 0 or self.b == 0:
            return

        H_infnorm = np.max(H_plus(H, self.n))
        self.norms.append(H_infnorm)
        number_samples = int(np.ceil(8 * epsilon ** (-2) * (np.log(2) * self.n + 5)))  # set log(1/p)=5
        if number_samples <= 0 or np.isnan(number_samples):
            raise f'Invalid sample number: {number_samples = }'

        self.cost += gibbs_cost(self.b, self.n, self.s, H_infnorm, epsilon / 2) * number_samples
        self.count_d += 1

    def sample_trArho(self, H, epsilon):
        if self.s == 0 or self.b == 0:
            return

        H_infnorm = np.max(H_plus(H, self.n))
        self.norms.append(H_infnorm)

        self.cost += trArho_cost(self.b, self.n, self.s, H_infnorm, epsilon)
        self.count_c += 1
