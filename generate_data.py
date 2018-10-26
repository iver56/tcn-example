import math
import os
import random

import joblib
import numpy as np


def generate_sine_series(n=200):
    offset = random.random() * 2 * math.pi
    frequency = 2 + 5 * random.random()
    x_vectors = []
    y_vectors = []
    for i in range(n):
        angle = (offset + (2 * math.pi / n) * i * frequency) % (2 * math.pi)
        sample = math.sin(angle)
        x_vectors.append([random.random(), sample])
        y_vectors.append([angle])
    return x_vectors, y_vectors


if __name__ == "__main__":
    n_sequences = 500
    x_sequences = []
    y_sequences = []
    for i in range(n_sequences):
        x, y = generate_sine_series()
        x_sequences.append(x)
        y_sequences.append(y)

    x_sequences = np.array(x_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.float32)

    os.makedirs("data", exist_ok=True)
    joblib.dump(
        {"x_sequences": x_sequences, "y_sequences": y_sequences},
        os.path.join("data", "dataset.pkl"),
    )
    print("Stored dataset")
