import numpy as np
from const import g
from NFD33_scheme import NFD
from CWA_scheme import CWA
from datetime import datetime
from graphics import simple_graphic


def time_steps_count() -> int:
    return 250  # - пока что так, потом если понадобится с постоянным/непостоянным шагом по времени


def periodicalU(a: float, x0: np.ndarray, X: float):
    return a * np.sin(2 * np.pi * x0 / X + np.pi / 4)


def periodicalH(b: int, u: np.ndarray):
    return (u + b) ** 2 / (4 * g)


def init(x_start: int, x_end: int, X: int, n: int, a: int, b: int):
    x = np.linspace(x_start, x_end, n)
    u = periodicalU(a, x, X)
    h = periodicalH(b, u)
    q = h * u
    return h, q, u, x


def start_compact_sceme(x_start: int, x_end: int, X: int, n: int, a: int, b: int, time_steps: int, snapshot_times: list, target_time: list):
    r = 0.05

    hu, qu, uu, x = init(x_start, x_end, X, n, a, b)  # n-1
    # simple_graphic(hu, x, 0)
    hv, qv, uv = np.zeros(n), np.zeros(n), np.zeros(n)  # n
    hw, qw, uw = np.zeros(n), np.zeros(n), np.zeros(n)  # n+1

    # time_steps = time_steps_count()
    trg_time = 0

    for t in range(time_steps + 1):
        if t == 0:
            hv, qv, uv = NFD(hu, qu, r)
        elif t in snapshot_times:
            np.save(f"./snapshots/h_T={target_time[trg_time]}_n={n}_const_dt", hw)
            np.save(f"./snapshots/q_T={target_time[trg_time]}_n={n}_const_dt", qw)
            print(f"Snapshot ZZZapisan: T={target_time[trg_time]}_n={n}_const_dt")
            trg_time += 1
            # simple_graphic(hw, x, t)

        else:
            hw, qw, uw = CWA(hv, qv, uv, hu, qu, uu, r, n)
            hu, qu, uu = hv, qv, uv
            hv, qv, uv = hw, qw, uw


def three_greeds():
    X = 10
    a = 2
    b = 10
    x_start = 0
    x_end = x_start + X

    N = [1001, 2001, 4001]
    T = [0.5, 1, 2.5]
    r = 0.05
    delta_h = [X / (i - 1) for i in N]
    delta_t = [dh * 0.05 for dh in delta_h]  # dt для каждой из сеток
    time_steps = [int(2.5 / dt) for dt in delta_t]  # количество шагов по времени для кажой из сеток

    snapshot_times_1 = [t / delta_t[0] for t in T]
    snapshot_times_2 = [t / delta_t[1] for t in T]
    snapshot_times_4 = [t / delta_t[2] for t in T]

    t1 = datetime.now()
    start_compact_sceme(x_start, x_end, X, N[0], a, b, time_steps[0], snapshot_times_1, T)
    t2 = datetime.now()
    print(f"Count CWA on {N[0]}-dot grid is {t2 - t1}")

    t1 = datetime.now()
    start_compact_sceme(x_start, x_end, X, N[1], a, b, time_steps[1], snapshot_times_2, T)
    t2 = datetime.now()
    print(f"Count CWA on {N[1]}-dot grid is {t2 - t1}")

    t1 = datetime.now()
    start_compact_sceme(x_start, x_end, X, N[2], a, b, time_steps[2], snapshot_times_4, T)
    t2 = datetime.now()
    print(f"Count CWA on {N[2]}-dot grid is {t2 - t1}")
