from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.layers.core import Activation
from keras.models import Sequential
from keras.optimizers import RMSprop


def get_bilstm_model(input_vector_size, output_vector_size):
    # Create model
    model = Sequential()
    model.add(
        TimeDistributed(
            Dense(32, activation="relu"), input_shape=(None, input_vector_size)
        )
    )
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(
        Bidirectional(
            LSTM(units=32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
        )
    )
    model.add(Activation("relu"))
    model.add(
        Bidirectional(
            LSTM(units=32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
        )
    )
    model.add(Activation("relu"))
    model.add(TimeDistributed(Dense(output_vector_size, activation="relu")))
    model.compile(
        loss="mse", optimizer=RMSprop()
    )
    model.summary()
    return model
