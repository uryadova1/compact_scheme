import numpy as np
from const import g
from sweep_methods import periodical_sweep_method


def norm(hv, hw, qv, qw):
    gg = (hv ** 2 - hw ** 2) ** (1 / 2) + (qv ** 2 - qw ** 2) ** (1 / 2)
    return gg


def create_matrix(vector, N):
    test_array = np.zeros((N, N))
    for i, value in enumerate(vector):
        if i == 0:
            test_array[i][i] = 4
            test_array[i][i + 1] = (1 + vector[i + 1])
            test_array[i][-1] = (1 - vector[- 1])
        elif i == N - 1:
            test_array[i][i] = 4
            test_array[i][i - 1] = (1 - vector[i - 1])
            test_array[i][0] = (1 + vector[0])
        else:
            test_array[i][i] = 4
            test_array[i][i + 1] = (1 + vector[i + 1])
            test_array[i][i - 1] = (1 - vector[i - 1])


def complex_vector_for_q_right_part(q, u, h):
    return q * u + h ** 2 * g / 2


def periodic_xxxx(uu, N):
    return np.concatenate([[uu[N - 2]], [uu[N - 1]], uu, [uu[1]], [uu[2]]])


def art_vis(vector_old, N):  # - поменять
    vector_old = periodic_xxxx(vector_old, N)
    Chq = 1.0 / 96.0
    artificial_viscosity = Chq
    delta_4x = 12 * artificial_viscosity * (
            vector_old[4:] - 4 * vector_old[3:-1] + 6 * vector_old[2:-2] - 4 * vector_old[1:-3] + vector_old[:-4])
    return delta_4x


def periodic(u, N):
    return np.concatenate([[u[N - 1]], u, [u[1]]])


def simple_sum_for_right_part(vec, N):
    vect = periodic(vec, N)
    return vect[:-2] + 4 * vect[1:-1] + vect[2:]


def sum_for_h_wk_1(vec, N):
    vect = periodic(vec, N)
    return vect[:-2] + vect[1:-1] + vect[2:]


def sum_for_right_part(vec, N):
    vec = periodic(vec, N)
    return vec[2:] - vec[:-2]


def right_part(vect_u, vect_f_u, vect_f_v, r, N):
    return sum_for_h_wk_1(vect_u, N) - r * (
            4 * sum_for_right_part(vect_f_v, N) + sum_for_right_part(vect_f_u, N)) - art_vis(vect_u, N)


def CWA(hv, qv, uv, hu, qu, uu, r, N):
    k = 0
    flag = True
    eps = 10 ** (-9)

    right_part_h = right_part(hu, qv, qu, r, N)

    vect_qu = complex_vector_for_q_right_part(qu, uu, hu)
    vect_qv = complex_vector_for_q_right_part(qv, uv, hv)
    right_part_q = right_part(qu, vect_qu, vect_qv, r, N)  # - g * r(
    # sum_for_right_part(hv, N))/2  # - перенесла в правую часть rg(h^(n+1)_(j+1) -h^(n+1)_(j-1))/2

    h_wk = hv
    q_wk = qv
    u_wk = uv

    while flag:
        # print(k)

        # h_wk_1 = periodical_sweep_method(create_matrix(u_wk[:-1], N), right_part_h[:-1], N)
        h_wk_1 = periodical_sweep_method(u_wk, right_part_h, r, N)
        # h_wk_1 = periodic(h_wk_1, N)

        right_part_q -= g * r * sum_for_h_wk_1(h_wk_1, N) / 2

        q_wk_1 = periodical_sweep_method(u_wk, right_part_q, r, N)

        if norm(h_wk, h_wk_1, q_wk, q_wk_1).all() < eps or k > 1000:  # - пока бредик, потому что это все таки разные случаи
            flag = False
        k += 1

        u_wk_1 = q_wk_1 / h_wk_1

        h_wk = h_wk_1
        q_wk = q_wk_1
        u_wk = u_wk_1
    print("goida")

    return h_wk, q_wk, u_wk
