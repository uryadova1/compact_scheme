import numpy as np


def get_true_index(cond):
    return np.array([i for i, v in enumerate(cond) if v])

def local_orders(h1, h2, h3, q1, q2, q3):
    # print(h1[0], h2[0], h3[0])
    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)[::2]
    h3 = np.asarray(h3, dtype=float)[::4]
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)[::2]
    q3 = np.asarray(q3, dtype=float)[::4]

    norm1 = ((h1 - h2) ** 2 + (q1 - q2) ** 2) ** (1 / 2)
    norm2 = ((h2 - h3) ** 2 + (q2 - q3) ** 2) ** (1 / 2)

    try:
        p = np.log2(np.array(abs(norm1 / norm2)))
        return p
    except TypeError:
        print("omg type error (opat'........)")


def relative_errors(h1, h2, h3, q1, q2, q3):
    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)[::2]
    h3 = np.asarray(h3, dtype=float)[::4]
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)[::2]
    q3 = np.asarray(q3, dtype=float)[::4]

    norm1 = ((h1 - h2) ** 2 + (q1 - q2) ** 2) ** (1 / 2)
    norm2 = ((h2 - h3) ** 2 + (q2 - q3) ** 2) ** (1 / 2)

    denom = (h1 ** 2 + q1 ** 2) ** (1 / 2)
    # denom = np.sqrt(np.power(h1, 2) + np.power(q1, 2))


    dw = abs(norm2 / norm1)
    epsilon = 1 / 2
    condition = np.greater_equal(dw, 1 - epsilon)
    cond = get_true_index(condition)

    if cond.size == 0:
        dw = 1 - epsilon
    else:
        dw[cond] = 1 - epsilon

    nom = norm1 / (1 - dw)
    return np.log10(nom / denom)