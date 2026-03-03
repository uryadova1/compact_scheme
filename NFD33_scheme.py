import numpy as np

from const import *


# переписать для 3

def L(l):  # np.array
    # l = np.concatenate((l[-3:-1], l, l[1:3]))
    return -((-1 / 12) * (l[4:] - l[:-4]) + (2 / 3) * (l[3:-1] - l[1:-3]))
    # return -a3 * l[:-6] - a2 * l[1:-5] - a1 * l[2:-4] + a1 * l[4:-2] + a2 * l[5:-1] + a3 * l[6:]


def artificial_viscosity(vector_old, art_vis):
    vector_old = np.concatenate((vector_old[-3: -1], vector_old, vector_old[1: 3]))
    return -(((vector_old[4:] + vector_old[:-4]) - 4 * (vector_old[3:-1] + vector_old[1:-3]) + 6 * vector_old[2:-2]) / art_vis)


def F(q, h):
    return np.array(q ** 2 / h + g * h ** 2 / 2)


def NFD(h_n, q_n, dt_dx):
    # step 1
    q_n_period = np.concatenate((q_n[-3: -1], q_n, q_n[1: 3]))
    h_n_period = np.concatenate((h_n[-3: -1], h_n, h_n[1:3]))

    l1 = L(q_n_period)
    l_arg_tmp = F(q_n_period, h_n_period)
    l2 = L(l_arg_tmp)

    h_1 = h_n + dt_dx * l1
    q_1 = q_n + dt_dx * l2

    # step 2

    q_1_period = np.concatenate((q_1[-3: -1], q_1, q_1[1: 3]))
    h_1_period = np.concatenate((h_1[-3: -1], h_1, h_1[1: 3]))

    l1 = L(q_1_period,)
    l_arg_tmp = F(q_1_period, h_1_period)
    l2 = L(l_arg_tmp)

    h_2 = (3 * h_n + h_1 + dt_dx * l1) / 4
    q_2 = (3 * q_n + q_1 + dt_dx * l2) / 4

    # step 3

    q_2_period = np.concatenate((q_2[-3: -1], q_2, q_2[1: 3]))
    h_2_period = np.concatenate((h_2[-3: -1], h_2, h_2[1: 3]))

    l1 = L(q_2_period)
    l_arg_tmp = F(q_2_period, h_2_period)
    l2 = L(l_arg_tmp)

    h_a = -artificial_viscosity(h_n, 16)
    q_a = -artificial_viscosity(q_n, 16)

    h_n_plus_1 = (h_n + 2 * h_2 + 2 * dt_dx * l1) / 3 + h_a
    q_n_plus_1 = (q_n + 2 * q_2 + 2 * dt_dx * l2) / 3 + q_a


    return h_n_plus_1, q_n_plus_1, q_n_plus_1/h_n_plus_1


def F_func(q, h):
    return np.array(q ** 2 / h + g * h ** 2 / 2)  # приведено к виду q^2/h+gh^2/2, чтобы не использовать u

def rusanov_scheme_p(h_n, q_n, R, C):
    # вот здесь периодические условия
    h_n = np.concatenate([[h_n[N - 3]], [h_n[N - 2]], h_n, [h_n[1]], [h_n[2]]])
    q_n = np.concatenate([[q_n[N - 3]], [q_n[N - 2]], q_n, [q_n[1]], [q_n[2]]])

    f_1 = F_func(q_n, h_n)  # задаем q^2/h+gh^2/2

    h_1 = (h_n[:-1] + h_n[1:]) / 2 - R * (q_n[1:] - q_n[:-1]) / 3  # n - 1
    q_1 = (q_n[:-1] + q_n[1:]) / 2 - R * (f_1[1:] - f_1[:-1]) / 3

    f_2 = F_func(q_1, h_1)

    h_2 = h_n[1:-1] - 2 / 3 * R * (q_1[1:] - q_1[:-1])
    q_2 = q_n[1:-1] - 2 / 3 * R * (f_2[1:] - f_2[:-1])  # n-2

    w_h = h_n[4:] - h_n[3:-1] * 4 + h_n[2:-2] * 6 - h_n[1:-3] * 4 + h_n[:-4]
    w_q = q_n[4:] - q_n[3:-1] * 4 + q_n[2:-2] * 6 - q_n[1:-3] * 4 + q_n[:-4]

    f_3 = F_func(q_2, h_2)
    h_3 = h_n[2:-2] - R * (7 / 24 * (q_n[3:-1] - q_n[1:-3]) - (2 / 24) * (q_n[4:] - q_n[:-4])) - 3 / 8 * R * (
            q_2[2:] - q_2[:-2]) - w_h * C / 24
    q_3 = q_n[2:-2] - R * ((7 / 24) * (f_1[3:-1] - f_1[1:-3]) - (2 / 24) * (f_1[4:] - f_1[:-4])) - 3 / 8 * R * (
            f_3[2:] - f_3[:-2]) - w_q * C / 24

    return h_3, q_3, q_3/h_3

