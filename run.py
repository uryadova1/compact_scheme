import numpy as np
from const import g
from NFD33_scheme import NFD
from CWA_scheme import CWA


def time_steps_count() -> int:
    pass


def periodicalU(a: float, x0: np.ndarray, X: float):
    return a * np.sin(2 * np.pi * x0 / X + np.pi / 4)


def periodicalH(b: int, u: np.ndarray):
    return (u + b) ** 2 / (4 * g)


def init(x_start, x_end, X, n, a, b):
    x = np.linspace(x_start, x_end, n)
    u = periodicalU(a, x, X)
    h = periodicalH(b, u)
    q = h * u
    return h, q, u


def start_compact_sceme():
    X = 10
    a = 2
    b = 10
    x_start = 0
    x_end = x_start + X
    n = 50

    # T = [0.5, 1, 2.5]

    hu, qu, uu = init(x_start, x_end, X, n, a, b)  # n-1
    hv, qv, uv = np.zeros(n), np.zeros(n), np.zeros(n)  # n
    hw, qw, qu = np.zeros(n), np.zeros(n), np.zeros(n)  # n+1

    time_steps = time_steps_count()

    for t in range(time_steps):
        if t == 0:
            hv, qv, uv = NFD(hu, qu, dt_dx=0.005)
        else:
            hu, qu, uu = hv, qv, uv
            hv, qv, uv = hw, qw, qu
            hw, qw, qu = CWA() #??????????????
