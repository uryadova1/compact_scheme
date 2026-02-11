import numpy as np

from const import g


# переписать для 3

def L(l):  # np.array
    l = np.concatenate((l[-3:-1], l, l[1:3]))
    return (-1 / 12) * (l[4:] - l[:-4]) + (2 / 3) * (l[3:-1] - l[1:-3])
    # return -a3 * l[:-6] - a2 * l[1:-5] - a1 * l[2:-4] + a1 * l[4:-2] + a2 * l[5:-1] + a3 * l[6:]


def artificial_viscosity(vector_old, art_vis):
    vector_old = np.concatenate((vector_old[-4: -1], vector_old, vector_old[1: 4]))
    new_vec = (vector_old[4:] - vector_old[:-4]) + - 4 * (vector_old[3:-1] - vector_old[1:-3]) + 6 * vector_old[2:-2]

    return new_vec / art_vis


def F(q, h):
    return np.array(q ** 2 / h + g * h ** 2 / 2)


def NFD(h_n, q_n, dt_dx):
    q_n_period = np.concatenate((q_n[-4: -1], q_n, q_n[1: 4]))
    h_n_period = np.concatenate((h_n[-4: -1], h_n, h_n[1: 4]))

    l1 = L(q_n_period)
    l_arg_tmp = F(q_n_period, h_n_period)
    l2 = L(l_arg_tmp)

    h_1 = h_n + dt_dx * l1
    q_1 = q_n + dt_dx * l2

    # step 2

    q_1_period = np.concatenate((q_1[-4: -1], q_1, q_1[1: 4]))
    h_1_period = np.concatenate((h_1[-4: -1], h_1, h_1[1: 4]))

    l1 = L(q_1_period)
    l_arg_tmp = F(q_1_period, h_1_period)
    l2 = L(l_arg_tmp,)

    h_2 = (3 * h_n + h_1 + dt_dx * l1) / 4
    q_2 = (3 * q_n + q_1 + dt_dx * l2) / 4

    # step 3

    q_2_period = np.concatenate((q_2[-4: -1], q_2, q_2[1: 4]))
    h_2_period = np.concatenate((h_2[-4: -1], h_2, h_2[1: 4]))

    l1 = L(q_2_period,)
    l_arg_tmp = F(q_2_period, h_2_period)
    l2 = L(l_arg_tmp)

    h_a = artificial_viscosity(h_n, 16)
    q_a = artificial_viscosity(q_n, 16)

    # f order == 4 or order == 8:
    h_a, q_a = -h_a, -q_a

    h_n_plus_1 = (h_n + 2 * h_2 + 2 * dt_dx * l1) / 3 + h_a
    q_n_plus_1 = (q_n + 2 * q_2 + 2 * dt_dx * l2) / 3 + q_a


    u_n_plus_1 = q_n_plus_1 / h_n_plus_1

    return h_n_plus_1, q_n_plus_1, u_n_plus_1
