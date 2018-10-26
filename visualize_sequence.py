import matplotlib.pyplot as plt

from generate_data import generate_sine_series

if __name__ == "__main__":
    x, y = generate_sine_series()
    fig, ax = plt.subplots()
    ax.plot(x)
    ax.plot(y)
    ax.grid()
    plt.show()
