import matplotlib.pyplot as plt

def simple_graphic(h, x):
    plt.figure(figsize=(10, 6))
    plt.plot(x, h, 'k o', markersize=0.8)
    plt.xlabel("x")
    plt.ylabel("h(x, t)")
    plt.grid()
    plt.show()