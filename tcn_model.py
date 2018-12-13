from keras.layers import Dense
from keras.models import Input, Model
from keras.optimizers import Adam

from tcn import TCN


def get_tcn_model(input_vector_size, target_vector_size):
    model_input = Input(shape=(None, input_vector_size))
    model_output = TCN(return_sequences=True, nb_filters=48)(model_input)
    model_output = Dense(target_vector_size, activation="relu")(model_output)

    model = Model(inputs=[model_input], outputs=[model_output])
    model.compile(optimizer=Adam(lr=0.002, clipnorm=1.0), loss="mse")

    model.summary()

    return model
