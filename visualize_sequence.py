import matplotlib.pyplot as plt
import numpy as np

from generate_data import generate_sine_series

if __name__ == "__main__":
    x, y = generate_sine_series()
    x = np.array(x)
    fig, ax = plt.subplots()
    ax.plot(x[:, 0], label='noise input')
    ax.plot(x[:, 1], label='sine input')
    ax.plot(y, label='target')
    ax.grid()
    plt.legend()
    plt.show()
