import numpy as np
from const import *
from NFD33_scheme import NFD, rusanov_scheme_periodical
from CWA_scheme import CWA
from datetime import datetime
from graphics import simple_graphic


def time_steps_count() -> int:
    return 250  # - пока что так, потом если понадобится с постоянным/непостоянным шагом по времени


def periodicalU(a: float, x0: np.ndarray, X: float):
    return a * np.sin(2 * np.pi * x0 / X + np.pi / 4)


def periodicalH(b: int, u: np.ndarray):
    return (u + b) ** 2 / (4 * g)


def init(n: int):
    x = np.linspace(x_start, x_end, n)
    u = periodicalU(a, x, X)
    h = periodicalH(b, u)
    q = h * u
    return h, q, u, x


def start_compact_sceme(N: int, time_steps: int, *args, **kwargs):
    r = 0.05

    snapshot_times = args[0] #kwargs.get("snapshot_times", None)
    target_time = args[1] #kwargs.get("target_time", None)

    length = len(snapshot_times)
    hu, qu, uu, x = init(N)  # n-1
    hv, qv, uv = np.zeros(N), np.zeros(N), np.zeros(N)  # n
    hw, qw, uw = np.zeros(N), np.zeros(N), np.zeros(N)  # n+1

    # trg_time = 0
    time_idx = 0

    for t in range(time_steps):
        if t == 0:
            hv, qv, uv = rusanov_scheme_periodical(hu, qu, 0.05, 0.104, N)
        else:
            hw, qw, uw = CWA(hv, qv, uv, hu, qu, uu, r, N)
            # if snapshot_times is not None and target_time is not None:
            if time_idx < length and t + 1 == snapshot_times[time_idx]:
                print(t)
                print(time_idx)
                np.save(f"./snapshots_GPU/h_T={target_time[time_idx]}_n={N}_const_dt", hw)
                np.save(f"./snapshots_GPU/q_T={target_time[time_idx]}_n={N}_const_dt", qw)
                print(f"Snapshot записан: T={target_time[time_idx]}_n={N}_const_dt")
                # trg_time += 1
                time_idx += 1

            hu, qu, uu = hv, qv, uv
            hv, qv, uv = hw, qw, uw



def check():
    T = 2.5
    delta_h = X / (101 - 1)
    delta_t = delta_h * 0.05
    time_steps = int(T / delta_t) + 1
    print(time_steps)
    start_compact_sceme(101, time_steps)



def three_greeds():
    N = [2 ** 10 + 1, 2 ** 11 + 1, 2**12 + 1] #- УЗЛЫ!!!!!
    T = [0.5, 1, 2.5]
    r = 0.05
    delta_h = [X / (i - 1) for i in N]
    delta_t = [dh * r for dh in delta_h]  # dt для каждой из сеток
    time_steps = [int(2.5 / dt) + 2 for dt in delta_t]  # количество шагов по времени для кажой из сеток

    snapshot_times_1 = [round(t / delta_t[0]) for t in T]
    snapshot_times_2 = [round(t / delta_t[1]) for t in T]
    snapshot_times_4 = [round(t / delta_t[2]) for t in T]

    total_start_time = datetime.now()
    t1 = datetime.now()
    print(f"Start compact scheme (GPU) on {N[0]} greed")
    start_compact_sceme(N[0], time_steps[0], snapshot_times_1, T)
    t2 = datetime.now()
    print(f"Count CWA on {N[0]}-dot grid is {t2 - t1}")

    t1 = datetime.now()
    print(f"Start compact scheme (GPU) on {N[1]} greed")
    start_compact_sceme(N[1], time_steps[1], snapshot_times_2, T)
    t2 = datetime.now()
    print(f"Count CWA on {N[1]}-dot grid is {t2 - t1}")

    t1 = datetime.now()
    print(f"Start compact scheme (GPU) on {N[2]} greed")
    start_compact_sceme(N[2], time_steps[2], snapshot_times_4, T)
    t2 = datetime.now()
    print(f"Count CWA on {N[2]}-dot grid is {t2 - t1}")
    total_end_time = datetime.now()
    print(f"\n Total time: {total_end_time - total_start_time}")


def easy_start():
    N = 401
    T = 1
    r = 0.05
    delta_h = X / (N - 1)
    delta_t = delta_h * r # dt для каждой из сеток
    time_steps = int(T / delta_t) + 2
    snapshot_times = [round(T / delta_t)]


    start_compact_sceme(N, time_steps, snapshot_times, [T])




