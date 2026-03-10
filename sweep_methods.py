import numpy as np


def thomas_algorithm(A, B, C, F, n):
    new_b = np.zeros(n, dtype=float)
    new_f = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

    new_b[0] = B[0]
    new_f[0] = F[0]

    for i in range(1, n):
        den = A[i] / new_b[i - 1]
        new_b[i] = B[i] - den * C[i - 1]
        new_f[i] = F[i] - den * new_f[i - 1]

    x[-1] = new_f[-1] / new_b[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (new_f[i] - C[i] * x[i + 1]) / new_b[i]

    return x


def periodical_sweep_method(vec, f, r, N):
    c = 1 + r * vec[1:]  # верхняя
    c = np.concatenate([c, [1 + r * vec[0]]])
    b = np.ones(N) * 4
    a = 1 - r * vec[:-1]  # нижняя - диагонали для матрицы An-1
    a = np.concatenate([[1 - r * vec[-1]], a])

    v_n_1 = np.concatenate(([c[-1]], np.zeros(N - 3), [a[-1]]))
    u_n_1 = np.concatenate(([a[0]], np.zeros(N - 3), [c[N - 2]]))

    c_n = b[-1]

    p_n_1 = thomas_algorithm(np.concatenate(([0], a[1:-1])), b[:-1], np.concatenate((c[:-2], [0])), f[:-1], N - 1)
    q_n_1 = thomas_algorithm(np.concatenate(([0], a[1:-1])), b[:-1], np.concatenate((c[:-2], [0])), u_n_1, N - 1)

    xn = (f[-1] - v_n_1 @ p_n_1) / (c_n - v_n_1 @ q_n_1)
    x_n_1 = p_n_1 - q_n_1 * xn
    return np.concatenate((x_n_1, [xn], [x_n_1[0]]))
