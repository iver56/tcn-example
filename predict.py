import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from generate_data import generate_sine_series

if __name__ == "__main__":

    model = load_model(os.path.join('data', 'tcn.h5'))

    x, y = generate_sine_series()
    x_np = np.array([x])
    y_np = np.array([y])

    y_predicted = model.predict(x_np).flatten()

    fig, ax = plt.subplots()
    ax.plot(x)
    ax.plot(y)
    ax.plot(y_predicted)
    ax.grid()
    plt.show()
