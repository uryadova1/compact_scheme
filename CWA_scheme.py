import numpy as np
from const import g
from sweep_methods import periodical_sweep_method
from sweep_methods_GPU import periodical_sweep_gpu


def norm1(hw, hw_1, qw, qw_1):
    return np.fabs(hw - hw_1) + np.fabs(qw - qw_1)


def F(q, h):
    return q ** 2 / h + h ** 2 * g / 2


def periodic_xx(vect):
    return np.concatenate([[vect[-3]], [vect[-2]], vect, [vect[1]], [vect[2]]])


def periodic_x(vect):
    return np.concatenate([[vect[-2]], vect, [vect[1]]])


def art_vis(vector_old):
    vector_old = periodic_xx(vector_old)
    artificial_viscosity = 1.0 / 96.0
    delta_4x = 12 * artificial_viscosity * (
            vector_old[4:] - 4 * vector_old[3:-1] + 6 * vector_old[2:-2] - 4 * vector_old[1:-3] + vector_old[:-4])
    return delta_4x


def simple_sum_for_n_minus_1_layer(vect):
    vect = periodic_x(vect)
    return vect[:-2] + 4 * vect[1:-1] + vect[2:]


def sum_for_right_part(vect):
    vect = periodic_x(vect)
    return vect[2:] - vect[:-2]


def right_part(vect_u, vect_f_u, vect_f_v, r):
    return simple_sum_for_n_minus_1_layer(vect_u) - 4 * r * sum_for_right_part(vect_f_v) - r * sum_for_right_part(
        vect_f_u) - art_vis(vect_u)


def CWA(hv, qv, uv, hu, qu, uu, r, N):
    k = 0
    flag = True
    eps = 10 ** (-9)

    right_part_h = right_part(hu, qu, qv, r)

    vect_qu = F(qu, hu)
    vect_qv = F(qv, hv)
    right_part_q = right_part(qu, vect_qu, vect_qv, r)

    h_wk = hv
    q_wk = qv
    u_wk = uv

    while flag:
        # h_wk_1 = periodical_sweep_method(u_wk[:-1], right_part_h[:-1], r, N - 1)
        h_wk_1 = periodical_sweep_gpu(u_wk[:-1], right_part_h[:-1], r, N - 1)

        right_part_q_add = right_part_q - g * r * sum_for_right_part(h_wk_1 ** 2) / 2

        # q_wk_1 = periodical_sweep_method(u_wk[:-1], right_part_q_add[:-1], r, N - 1)
        q_wk_1 = periodical_sweep_gpu(u_wk[:-1], right_part_q_add[:-1], r, N - 1)

        u_wk_1 = q_wk_1 / h_wk_1

        if norm1(h_wk, h_wk_1, u_wk, u_wk_1).all() < eps:
            # print(h_wk_1)
            # exit(88)
            # print(k)
            return h_wk_1, q_wk_1, q_wk_1 / h_wk_1

        elif k > 1000:
            exit(67)
        k += 1

        h_wk = h_wk_1
        q_wk = q_wk_1
        u_wk = u_wk_1
    return h_wk, q_wk, u_wk
