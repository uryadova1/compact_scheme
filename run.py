import numpy as np
from const import g
from NFD33_scheme import NFD
from CWA_scheme import CWA
from graphics import simple_graphic


def time_steps_count() -> int:
    return 50 # - пока что так, потом если понадобится с постоянным/непостоянным шагом по времени


def periodicalU(a: float, x0: np.ndarray, X: float):
    return a * np.sin(2 * np.pi * x0 / X + np.pi / 4)


def periodicalH(b: int, u: np.ndarray):
    return (u + b) ** 2 / (4 * g)


def init(x_start, x_end, X, n, a, b):
    x = np.linspace(x_start, x_end, n)
    u = periodicalU(a, x, X)
    h = periodicalH(b, u)
    q = h * u
    return h, q, u, x


def start_compact_sceme():
    X = 10
    a = 2
    b = 10
    x_start = 0
    x_end = x_start + X
    n = 51

    # delta_h = X / (n - 1) 0.2
    # delta_t = delta_h * 0.05 0.01
    r = 0.05

    # T = [0.5, 1, 2.5]

    hu, qu, uu, x = init(x_start, x_end, X, n, a, b)  # n-1
    # simple_graphic(hu, x)
    hv, qv, uv = np.zeros(n), np.zeros(n), np.zeros(n)  # n
    hw, qw, uw = np.zeros(n), np.zeros(n), np.zeros(n)  # n+1

    time_steps = time_steps_count()

    for t in range(time_steps):
        if t == 0:
            hv, qv, uv = NFD(hu, qu, 0.05)
            # simple_graphic(hv, x)
        else:
            hw, qw, uw = CWA(hv, qv, uv, hu, qu, uu, r, n) #??????????????
            hu, qu, uu = hv, qv, uv
            hv, qv, uv = hw, qw, qu
            # simple_graphic(hw, x)
    simple_graphic(hw, x)