import os

import joblib
from keras.layers import Dense
from keras.models import Input, Model

from tcn import TCN

if __name__ == '__main__':
    data = joblib.load(os.path.join('data', 'dataset.pkl'))
    x_sequences = data['x_sequences']
    y_sequences = data['y_sequences']

    n_timesteps = len(x_sequences[0])
    input_vector_size = len(x_sequences[0][0])
    target_vector_size = len(y_sequences[0][0])

    model_input = Input(shape=(None, input_vector_size))
    model_output = TCN(return_sequences=True)(model_input)
    model_output = Dense(target_vector_size, activation='relu')(model_output)

    model = Model(inputs=[model_input], outputs=[model_output])
    model.compile(optimizer='adam', loss='mse')

    model.summary()

    model.fit(x_sequences, y_sequences, epochs=10, validation_split=0.2)

    model.save(os.path.join('data', 'tcn.h5'))

    print('Model is saved')
