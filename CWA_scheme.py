import numpy as np
from const import g
from sweep_methods import periodical_sweep_method


def norm1(hw, hw_1, qw, qw_1):
    return np.fabs(hw - hw_1) + np.fabs(qw - qw_1)

    # return (abs(hw ** 2 - hw_1 ** 2)) ** (1 / 2) + (abs(qw ** 2 - qw_1 ** 2)) ** (1 / 2)


def norm(vect_w, vect_w_1):
    gg = (abs(vect_w ** 2 - vect_w_1 ** 2)) ** (1 / 2)
    return gg


def F(q, u, h):
    return q * u + h ** 2 * g / 2


def periodic_xx(vect, N):
    return np.concatenate([[vect[-2]], [vect[-1]], vect, [vect[1]], [vect[2]]])


def periodic_x(vect, N):
    return np.concatenate([[vect[-1]], vect, [vect[1]]])


def art_vis(vector_old, N):
    vector_old = periodic_xx(vector_old, N)
    artificial_viscosity = 1.0 / 96.0
    delta_4x = 12 * artificial_viscosity * (
            vector_old[4:] - 4 * vector_old[3:-1] + 6 * vector_old[2:-2] - 4 * vector_old[1:-3] + vector_old[:-4])
    return delta_4x


def simple_sum_for_right_part(vect, N):
    vect = periodic_x(vect, N)
    return vect[:-2] + 4 * vect[1:-1] + vect[2:]


def sum_for_right_part(vect, N):
    vect = periodic_x(vect, N)
    return vect[2:] - vect[:-2]


def right_part(vect_u, vect_f_u, vect_f_v, r, N):
    return simple_sum_for_right_part(vect_u, N) - 4 * r * sum_for_right_part(vect_f_v, N) - r * sum_for_right_part(
        vect_f_u, N) - art_vis(vect_u, N)


def CWA(hv, qv, uv, hu, qu, uu, r, N):
    k = 0
    flag = True
    eps = 10 ** (-8)

    right_part_h = right_part(hu, qu, qv, r, N)

    vect_qu = F(qu, uu, hu)
    vect_qv = F(qv, uv, hv)
    right_part_q = right_part(qu, vect_qu, vect_qv, r, N)

    h_wk = hv
    q_wk = qv
    u_wk = uv

    while flag:
        h_wk_1 = periodical_sweep_method(u_wk, right_part_h, r, N)

        right_part_q_refresh = right_part_q - g * r * sum_for_right_part(h_wk_1 ** 2, N) / 2

        q_wk_1 = periodical_sweep_method(u_wk, right_part_q_refresh, r, N)

        if norm1(h_wk, h_wk_1, q_wk, q_wk_1).all() < eps:
            flag = False
            # print(k)
        elif k > 1000:
            break
        k += 1

        u_wk_1 = q_wk_1 / h_wk_1

        h_wk = h_wk_1
        q_wk = q_wk_1
        u_wk = u_wk_1

    return h_wk, q_wk, u_wk
