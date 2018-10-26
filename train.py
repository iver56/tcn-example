import os

import joblib

from bilstm_model import get_bilstm_model
from tcn_model import get_tcn_model

if __name__ == '__main__':
    data = joblib.load(os.path.join('data', 'dataset.pkl'))
    x_sequences = data['x_sequences']
    y_sequences = data['y_sequences']

    n_timesteps = len(x_sequences[0])
    input_vector_size = len(x_sequences[0][0])
    target_vector_size = len(y_sequences[0][0])

    model = get_tcn_model(input_vector_size, target_vector_size)
    #model = get_bilstm_model(input_vector_size, target_vector_size)

    model.fit(x_sequences, y_sequences, epochs=10, validation_split=0.2)

    model.save(os.path.join('data', 'tcn.h5'))

    print('Model is saved')
