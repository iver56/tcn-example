from keras.layers import Dense
from keras.models import Input, Model
from keras.optimizers import Adam

from tcn import TCN


def get_tcn_model(
    input_vector_size, target_vector_size, num_filters=48, learning_rate=0.002
):
    model_input = Input(shape=(None, input_vector_size))
    model_output = TCN(return_sequences=True, nb_filters=int(num_filters))(model_input)
    model_output = Dense(target_vector_size, activation="relu")(model_output)

    model = Model(inputs=[model_input], outputs=[model_output])
    model.compile(optimizer=Adam(lr=learning_rate, clipnorm=1.0), loss="mse")

    # model.summary()

    return model
