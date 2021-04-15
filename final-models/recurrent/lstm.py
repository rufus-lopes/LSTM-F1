import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import sqlite3
import pandas as pd
from pathlib import Path
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional

physical_devices = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def training_data():

    dir = '../../../SQL_Data/constant_setup'
    files = os.listdir(dir)
    files = [f for f in files if f.endswith('.sqlite3')]

    data = []
    for f in files:
        path = os.path.join(dir, f)
        conn = sqlite3.connect(path)
        if os.path.getsize(path) > 10000:
            cur = conn.cursor()
            cur.execute('SELECT * FROM TrainingData')
            df = pd.DataFrame(cur.fetchall())
            data.append(df)

    names = list(map(lambda x: x[0], cur.description))
    df = pd.concat(data)
    df.columns = names
    df = df.drop(['frameIdentifier','bestLapTime', 'pkt_id', 'packetId', 'SessionTime', 'lap_time_remaining'], axis=1)
    df.set_index('index', inplace=True)

    return df

def pad_data(training, target):

    max_timesteps = 10000 # max(training, key=len).shape[0]
    num_rows_to_add = [max_timesteps-l.shape[0] for l in training]
    training_pad = []
    target_pad = []
    print(f'max timesteps : {max_timesteps}')

    for i in range(len(training)):
        rows_to_add = num_rows_to_add[i]

        training_arr = training[i]
        training_append = np.zeros((rows_to_add, training[0].shape[1]), dtype=float)
        training_array = np.vstack((training_arr, training_append))
        training_pad.append(training_array)

        target_arr = target[i].reshape(target[i].shape[0])
        target_append = np.zeros((rows_to_add), dtype=float)
        target_array = np.concatenate([target_arr, np.zeros(rows_to_add)])
        target_pad.append(target_array)

    return training_pad, target_pad

def scale_data(data):
    scalers = {}
    sessionUIDs = data.pop('sessionUID')
    lap_number = data.pop('currentLapNum')
    for i in data.columns:
        scaler = MinMaxScaler()
        s = scaler.fit_transform(data[i].values.reshape(-1,1))
        s = np.reshape(s, len(s))
        scalers['scaler_'+ i ] = scaler
        data[i] = s

    data['sessionUID'] = sessionUIDs
    data['currentLapNum'] = lap_number

    return data, scalers

def format_data(data):
    '''

    seperates data first by session, then by lap, before padding each array so that
    they are all the same length for model input.
    Performs test train split also

    '''
    data.reset_index(drop=True, inplace=True)

    session_groups = data.groupby('sessionUID')
    training_data = []
    target_data = []
    total_laps = 0
    for s in list(session_groups.groups):
        session = session_groups.get_group(s)
        lap_groups = session.groupby('currentLapNum')
        total_laps += len(lap_groups)
        for l in list(lap_groups.groups):
            lap = lap_groups.get_group(l)
            lap = lap.drop(['sessionUID'], axis=1)
            target_data.append(lap.pop('finalLapTime'))
            training_data.append(lap)

    training = [x.to_numpy() for x in training_data]
    target = [y.to_numpy() for y in target_data]
    print(f'Total Laps: {total_laps}')

    max_timesteps = 10000 # max(training, key=len).shape[0]
    num_rows_to_add = [max_timesteps-l.shape[0] for l in training]

    training_pad = []
    target_pad = []
    print(f'max timesteps : {max_timesteps}')

    for i in range(len(training)):
        rows_to_add = num_rows_to_add[i]

        training_arr = training[i]
        training_append = np.zeros((rows_to_add, training[0].shape[1]), dtype=float)
        training_array = np.vstack((training_arr, training_append))
        training_pad.append(training_array)

        target_arr = target[i]
        target_append = np.zeros((rows_to_add), dtype=float)
        target_array = np.concatenate([target_arr, np.zeros(rows_to_add)])
        target_pad.append(target_array)

    split = int(total_laps*0.9)

    X_train = training_pad[:split]
    X_test = training_pad[split:]
    y_train = target_pad[:split]
    y_test = target_pad[split:]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_train = np.array(y_train)

    print('Data Formatted')

    return X_train, X_test, y_train, y_test

def train_model(model, trainX, trainY):
    ''' training the model'''
    EPOCHS = 100
    callback = [EarlyStopping(monitor="loss", patience = 10, mode = 'auto',
                restore_best_weights=True),
                ModelCheckpoint('generator_lstm.h5')]
    history = model.fit(trainX, trainY, callbacks=callback, shuffle=False, epochs=EPOCHS, batch_size=1)
    return history, model

def build_model(trainX):

    ''' buils model and prints out summary'''

    learning_rate = 0.001
    units = 128
    epochs = 100
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU
    from tensorflow.keras.callbacks import EarlyStopping

    model = Sequential()
    model.add(Bidirectional(LSTM(units, return_sequences=True, input_shape=(None, trainX.shape[2]) )))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units, return_sequences=True)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    adam = tf.keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    print('Model Built')

    return model

def plot_residuals(testX, testY, pred, data):
    residuals = []
    predictions = pred.reshape(pred.shape[0], pred.shape[1])
    d = data.drop(['sessionUID', 'finalLapTime'], axis=1)
    names = d.columns
    err = testY-predictions
    for i in range(len(predictions)):
        test_X = pd.DataFrame(testX[i], columns = names)
        test_X['predictions'] = predictions[i]
        test_X['residuals'] = err[i]
        residuals.append(pd.DataFrame(test_X))
    final = pd.concat(residuals)
    return final



if __name__ == '__main__':
    data = training_data()

    # data = pd.read_csv('../../Bayesian_RNN/sample_data.csv')
    # data.drop(['index'], axis=1, inplace=True)

    data.reset_index(drop=True, inplace=True)

    trainX, testX, trainY, testY = format_data(data)

    # model = build_model(trainX)
    #
    # history, model = train_model(model, trainX, trainY)

    model = tf.keras.models.load_model('generator_lstm.h5')
    print(f'test X shape {testX.shape}')

    predictions = model.predict(testX)

    print(f'test Y shape {testY.shape}')
    y_con = pd.concat(testY)
    print(y_con)
    print(y_con.shape)
    test_dataframe = plot_residuals(testX, testY, predictions, data)

    x = test_dataframe['currentLapTime']
    y = test_dataframe['residuals']

    test_dataframe.to_csv('LSTM_unscaled_predictions.csv')

    plt.scatter(x,y, s=5, alpha=0.5)
    plt.show()

    x = test_dataframe['currentLapTime']
    y = test_dataframe['predictions']

    plt.scatter(x,y, s=0.05,)
    plt.show()
