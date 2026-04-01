import matplotlib.pyplot as plt
import numpy as np
from count_data import relative_errors, local_orders


def simple_graphic(h, x, t):
    plt.figure(figsize=(10, 6))
    plt.plot(x, h, 'k o', markersize=0.8, label=f"T = {t}")
    plt.xlabel("x")
    plt.ylabel("h(x, t)")
    plt.legend()
    plt.grid()
    plt.show()


def relative_errors_and_local_orders_graphic():
    X = 10
    x_start = 0
    x_end = x_start + X
    n = 101
    x_lin = np.linspace(x_start, x_end, n)

    path_part = './snapshots_GPU/'  # './snapshots/'

    h_1_05 = np.load(path_part + "h_T=1_n=1001_const_dt.npy", allow_pickle=True)
    q_1_05 = np.load(path_part + "q_T=1_n=1001_const_dt.npy", allow_pickle=True)

    h_2_05 = np.load(path_part + "h_T=1_n=2001_const_dt.npy", allow_pickle=True)
    q_2_05 = np.load(path_part + "q_T=1_n=2001_const_dt.npy", allow_pickle=True)

    h_4_05 = np.load(path_part + "h_T=1_n=4001_const_dt.npy", allow_pickle=True)
    q_4_05 = np.load(path_part + "q_T=1_n=4001_const_dt.npy", allow_pickle=True)

    h_1_1 = np.load(path_part + "h_T=1_n=1001_const_dt.npy", allow_pickle=True)
    q_1_1 = np.load(path_part + "q_T=1_n=1001_const_dt.npy", allow_pickle=True)

    h_2_1 = np.load(path_part + "h_T=1_n=2001_const_dt.npy", allow_pickle=True)
    q_2_1 = np.load(path_part + "q_T=1_n=2001_const_dt.npy", allow_pickle=True)

    h_4_1 = np.load(path_part + "h_T=1_n=4001_const_dt.npy", allow_pickle=True)
    q_4_1 = np.load(path_part + "q_T=1_n=4001_const_dt.npy", allow_pickle=True)

    h_1_25 = np.load(path_part + "h_T=2.5_n=1001_const_dt.npy", allow_pickle=True)
    q_1_25 = np.load(path_part + "q_T=2.5_n=1001_const_dt.npy", allow_pickle=True)

    h_2_25 = np.load(path_part + "h_T=2.5_n=2001_const_dt.npy", allow_pickle=True)
    q_2_25 = np.load(path_part + "q_T=2.5_n=2001_const_dt.npy", allow_pickle=True)

    h_4_25 = np.load(path_part + "h_T=2.5_n=4001_const_dt.npy", allow_pickle=True)
    q_4_25 = np.load(path_part + "q_T=2.5_n=4001_const_dt.npy", allow_pickle=True)

    relative_errors_1 = relative_errors(h_1_05, h_2_05, h_4_05, q_1_05, q_2_05, q_4_05)
    relative_errors_2 = relative_errors(h_1_1, h_2_1, h_4_1, q_1_1, q_2_1, q_4_1)
    relative_errors_4 = relative_errors(h_1_25, h_2_25, h_4_25, q_1_25, q_2_25, q_4_25)

    local_orders_1 = local_orders(h_1_05, h_2_05, h_4_05, q_1_05, q_2_05, q_4_05)
    local_orders_2 = relative_errors(h_1_1, h_2_1, h_4_1, q_1_1, q_2_1, q_4_1)
    local_orders_4 = relative_errors(h_1_25, h_2_25, h_4_25, q_1_25, q_2_25, q_4_25)

    plt.figure(figsize=(10, 6))
    plt.plot(x_lin, h_1_05, 'k o', markersize=0.8, label="H")
    plt.plot(x_lin, relative_errors_1, label="CWA")
    plt.title("T = 0.5")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Relative errors")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_lin, h_1_1, 'k o', markersize=0.8, label="H")
    plt.plot(x_lin, relative_errors_2, label="CWA")
    plt.title("T = 1.0")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Relative errors")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_lin, h_1_25, 'k o', markersize=0.8, label="H")
    plt.plot(x_lin, relative_errors_4, label="CWA")
    plt.title("T = 2.5")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Relative errors")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_lin, h_1_05, 'k o', markersize=0.8, label="H")
    plt.plot(x_lin, local_orders_1, label="CWA")
    plt.title("T = 0.5")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Local orders")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_lin, h_1_1, 'k o', markersize=0.8, label="H")
    plt.plot(x_lin, local_orders_2, label="CWA")
    plt.title("T = 1.0")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Local orders")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_lin, h_1_1, 'k o', markersize=0.8, label="H")
    plt.plot(x_lin, local_orders_4, label="CWA")
    plt.title("T = 2.5")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Local orders")
    plt.grid()
    plt.show()
