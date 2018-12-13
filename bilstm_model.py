from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.layers.core import Activation
from keras.models import Sequential
from keras.optimizers import RMSprop


def get_bilstm_model(
    input_vector_size,
    output_vector_size,
    dropout0=0.2,
    dropout1=0.1,
    recurrent_dropout0=0.1,
    dropout2=0.2,
    recurrent_dropout1=0.1,
    learning_rate=0.001,
):
    # Create model
    model = Sequential()
    model.add(
        TimeDistributed(
            Dense(32, activation="relu"), input_shape=(None, input_vector_size)
        )
    )
    model.add(TimeDistributed(Dropout(dropout0)))
    model.add(
        Bidirectional(
            LSTM(
                units=32,
                return_sequences=True,
                dropout=dropout1,
                recurrent_dropout=recurrent_dropout0,
            )
        )
    )
    model.add(Activation("relu"))
    model.add(
        Bidirectional(
            LSTM(
                units=32,
                return_sequences=True,
                dropout=dropout2,
                recurrent_dropout=recurrent_dropout1,
            )
        )
    )
    model.add(Activation("relu"))
    model.add(TimeDistributed(Dense(output_vector_size, activation="relu")))
    model.compile(loss="mse", optimizer=RMSprop(lr=learning_rate))
    # model.summary()
    return model
