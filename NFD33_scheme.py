import numpy as np

from const import g


# переписать для 3

def L(l):  # np.array
    l = np.concatenate((l[-3:-1], l, l[1:3]))
    return (-1 / 12) * (l[4:] - l[:-4]) + (2 / 3) * (l[3:-1] - l[1:-3])
    # return -a3 * l[:-6] - a2 * l[1:-5] - a1 * l[2:-4] + a1 * l[4:-2] + a2 * l[5:-1] + a3 * l[6:]


def artificial_viscosity(vector_old, art_vis):
    vector_old = np.concatenate((vector_old[-3: -1], vector_old, vector_old[1: 3]))
    new_vec = (vector_old[4:] - vector_old[:-4]) + 4 * (vector_old[3:-1] - vector_old[1:-3]) + 6 * vector_old[2:-2]

    return new_vec / art_vis


def F(q, h):
    return np.array(q ** 2 / h + g * h ** 2 / 2)


def NFD(h_n, q_n, dt_dx):
    # step 1
    l1 = L(q_n)
    l_arg_tmp = F(q_n, h_n)
    l2 = L(l_arg_tmp)

    h_1 = h_n + dt_dx * l1
    q_1 = q_n + dt_dx * l2

    # step 2
    l1 = L(q_1)
    l_arg_tmp = F(q_1, h_1)
    l2 = L(l_arg_tmp)

    h_2 = (3 * h_n + h_1 + dt_dx * l1) / 4
    q_2 = (3 * q_n + q_1 + dt_dx * l2) / 4

    # step 3
    l1 = L(q_2)
    l_arg_tmp = F(q_2, h_2)
    l2 = L(l_arg_tmp)

    h_a = -artificial_viscosity(h_n, 16)
    q_a = -artificial_viscosity(q_n, 16)

    # if order == 4 or order == 8:
    #     h_a, q_a = -h_a, -q_a

    h_n_plus_1 = (h_n + 2 * h_2 + 2 * dt_dx * l1) / 3 + h_a
    q_n_plus_1 = (q_n + 2 * q_2 + 2 * dt_dx * l2) / 3 + q_a

    return h_n_plus_1, q_n_plus_1, q_n_plus_1/h_n_plus_1
