import numpy as np


def thomas_algorithm(A, B, C, F, n):
    alpha = np.zeros(n + 1, dtype=float)
    beta = np.zeros(n + 1, dtype=float)
    x = np.zeros(n, dtype=float)

    alpha[1] = -C[0] / B[0]
    beta[1] = F[0] / B[0]

    for i in range(1, n - 1):
        den = A[i] * alpha[i] + B[i]
        alpha[i + 1] = -C[i] / den
        beta[i + 1] = (F[i] - A[i] * beta[i]) / den

    x[-1] = (F[-1] - A[-1] * beta[-1]) / (B[-1] + A[-1] * alpha[-1])
    # x[-1] = beta[-1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]

    return x


def periodical_sweep_method(vec, f, r, N):
    a = 1 + r * vec
    b = np.ones(N) * 4
    c = 1 - r * vec # - для матрицы An-1

    v_n_1 = np.concatenate(([b[-1]], np.zeros(N - 3), [a[-1]]))
    u_n_1 = np.concatenate(([a[0]], np.zeros(N - 3), [b[N - 2]]))


    c_n = c[-1]

    # gg = np.concatenate(([0], a[1:-1]))

    P_n_1 = thomas_algorithm(np.concatenate(([0], a[1:-1])), c[:-1], np.concatenate((b[:-2], [0])), f[:-1], N - 1)
    q_n_1 = thomas_algorithm(np.concatenate(([0], a[1:-1])), c[:-1], np.concatenate((b[:-2], [0])), u_n_1, N - 1)

    xn = (f[-1] - v_n_1 @ P_n_1) / (c_n - v_n_1 @ q_n_1)
    x_n_1 = P_n_1 - q_n_1 * xn
    return np.concatenate((x_n_1, [xn]))


