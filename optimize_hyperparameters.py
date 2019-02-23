import os

import joblib
import numpy as np
from hyperas import optim
from hyperopt import STATUS_OK, tpe, Trials
from hyperopt.pyll.stochastic import choice, quniform, uniform
from keras import backend as K
from sklearn.model_selection import train_test_split

from bilstm_model import get_bilstm_model
from tcn_model import get_tcn_model


def get_data():
    data = joblib.load(os.path.join("data", "dataset.pkl"))
    x_sequences = data["x_sequences"]
    y_sequences = data["y_sequences"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_sequences, y_sequences, test_size=0.2, random_state=42
    )

    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    input_vector_size = len(x_train[0][0])
    target_vector_size = len(y_train[0][0])

    model_type = {{choice(["lstm", "tcn"])}}
    learning_rate = {{uniform(0.001, 0.01)}}
    if model_type == "lstm":
        model = get_bilstm_model(
            input_vector_size,
            target_vector_size,
            dropout0={{uniform(0, 0.2)}},
            dropout1={{uniform(0, 0.1)}},
            recurrent_dropout0={{uniform(0, 0.1)}},
            dropout2={{uniform(0, 0.2)}},
            recurrent_dropout1={{uniform(0, 0.1)}},
            learning_rate=learning_rate,
        )
    else:
        model = get_tcn_model(
            input_vector_size,
            target_vector_size,
            num_filters={{quniform(16, 256, q=4)}},
            learning_rate=learning_rate,
        )

    result = model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test))

    best_validation_loss = np.amin(result.history["val_loss"])

    K.clear_session()

    return {"loss": best_validation_loss, "status": STATUS_OK}


if __name__ == "__main__":
    best_run, best_model = optim.minimize(
        model=create_model,
        data=get_data,
        algo=tpe.suggest,
        max_evals=15,
        trials=Trials(),
    )
    print("====")
    print("Recommended hyper-parameters:")
    print(best_run)
