import numpy as np
from const import *
from NFD33_scheme import NFD, rusanov_scheme_p
from CWA_scheme import CWA
from datetime import datetime
from graphics import simple_graphic
from try_k import compact_weak_approximation_scheme_SH


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


def start_compact_sceme(N: int, time_steps: int, snapshot_times: list, target_time: list): #N: int, time_steps: int, snapshot_times: list, target_time: list
    r = 0.05

    hu, qu, uu, x = init(N)  # n-1
    hv, qv, uv = np.zeros(N), np.zeros(N), np.zeros(N)  # n
    hw, qw, uw = np.zeros(N), np.zeros(N), np.zeros(N)  # n+1

    # time_steps = time_steps_count()
    trg_time = 0

    for t in range(time_steps):
        if t == 0:
            hv, qv, uv = rusanov_scheme_p(hu, qu, 0.05, 0.104, N) #NFD(hu, qu, r) #
        else:
            hw, qw, uw = CWA(hv, qv, uv, hu, qu, uu, r, N)
            if t in snapshot_times:
                print(t)
                np.save(f"./gg/h_T={target_time[trg_time]}_n={N}_const_dt", hw)
                np.save(f"./gg/q_T={target_time[trg_time]}_n={N}_const_dt", qw)
                print(f"Snapshot ZZZapisan: T={target_time[trg_time]}_n={N}_const_dt")
                trg_time += 1
            hu, qu, uu = hv, qv, uv
            hv, qv, uv = hw, qw, uw
    # print(hw)
    # simple_graphic(hw, x, time_steps)



def check():
    T = 2.5
    delta_h = X / (101 - 1)
    delta_t = delta_h * 0.05
    time_steps = int(T / delta_t) + 1
    print(time_steps)
    start_compact_sceme(101, time_steps)



def three_greeds():
    N = [101, 201, 401] #- УЗЛЫ!!!!!
    T = [0.5, 1, 2.5]
    r = 0.05
    delta_h = [X / (i - 1) for i in N]
    delta_t = [dh * r for dh in delta_h]  # dt для каждой из сеток
    time_steps = [int(2.5 / dt) + 2 for dt in delta_t]  # количество шагов по времени для кажой из сеток

    snapshot_times_1 = [round(t / delta_t[0]) for t in T]
    snapshot_times_2 = [round(t / delta_t[1]) for t in T]
    snapshot_times_4 = [round(t / delta_t[2]) for t in T]

    t1 = datetime.now()
    start_compact_sceme(N[0], time_steps[0], snapshot_times_1, T)
    t2 = datetime.now()
    print(f"Count CWA on {N[0]}-dot grid is {t2 - t1}")

    t1 = datetime.now()
    start_compact_sceme(N[1], time_steps[1], snapshot_times_2, T)
    t2 = datetime.now()
    print(f"Count CWA on {N[1]}-dot grid is {t2 - t1}")

    t1 = datetime.now()
    start_compact_sceme(N[2], time_steps[2], snapshot_times_4, T)
    t2 = datetime.now()
    print(f"Count CWA on {N[2]}-dot grid is {t2 - t1}")


def easy_start():
    N = 401
    T = 1
    r = 0.05
    delta_h = X / (N - 1)
    delta_t = delta_h * r # dt для каждой из сеток
    time_steps = int(T / delta_t) + 2
    snapshot_times = [round(T / delta_t)]


    start_compact_sceme(N, time_steps, snapshot_times, [T])




