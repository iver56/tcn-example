import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from generate_data import generate_sine_series

if __name__ == "__main__":

    model = load_model(os.path.join('data', 'tcn.h5'))

    x, y = generate_sine_series()
    x = np.array([x])

    y_predicted = model.predict(x).flatten()

    fig, ax = plt.subplots()
    ax.plot(x[0, :, 0], label='noise input')
    ax.plot(x[0, :, 1], label='sine input')
    ax.plot(y, label='ground truth')
    ax.plot(y_predicted, label='predicted')
    ax.grid()
    plt.legend()
    plt.show()
