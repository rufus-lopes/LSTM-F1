import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sqlite3
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

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

def pad_laps(data):
    frames = []
    sessions = data.groupby('sessionUID')
    for s in list(sessions.groups):
        session = sessions.get_group(s)
        laps = session.groupby('currentLapNum')
        for l in list(laps.groups):
            lap = laps.get_group(l)
            for i in range(5):
                lap = lap.append(pd.Series(0, index=lap.columns), ignore_index=True)
            frames.append(lap)
    frames = pd.concat(frames)
    return frames

def scale_data(data):
    training = pd.DataFrame()
    scalers = {}
    for i in data.columns:
        scaler = MinMaxScaler(feature_range=(-1,1))
        s = scaler.fit_transform(data[i].values.reshape(-1,1))
        s = np.reshape(s, len(s))
        scalers['scaler_'+ i ] = scaler
        training[i] = s
    return training, scalers

def generate_data(training, target):
    trainX, testX, trainY, testY = train_test_split(training, target, test_size=0.2, random_state=42, shuffle = False)


    test_sessions = testX.pop('sessionUID').to_numpy()
    train_session = trainX.pop('sessionUID')

    test_df_names = list(testX.columns)


    trainX = trainX.to_numpy()
    testX = testX.to_numpy()
    trainY = trainY.to_numpy()
    testY = testY.to_numpy()

    timesteps = training.shape[0]-1
    look_back = 5
    batch_size = 2
    train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(trainX, trainY, length=look_back, sampling_rate=1, stride=1, batch_size=1)
    test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(testX, testY, length=look_back, sampling_rate=1, stride=1, batch_size=1)

    return train_generator, test_generator, testX, testY, test_sessions

def build_model(trainX):
    learning_rate = 0.0001
    units = 64
    model = Sequential()
    model.add(Bidirectional(LSTM(units, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units=units, return_sequences=True)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(LSTM(units, return_sequences=True))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
    adam = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    return model

def train_model(trainX, trainY, model):
    callback = [EarlyStopping(monitor="loss", min_delta = 0.0001, patience = 3, mode = 'auto', restore_best_weights=True),
                ModelCheckpoint('normal_lstm.h5')]
    epochs = 100
    history = model.fit(x=trainX,y=trainY,
                        callbacks=callback,
                        shuffle=False,
                        epochs=epochs,
                        batch_size=1)
    return model, history



def plot_residuals(testX, testY, pred, data):
    data.drop(['sessionUID', 'finalLapTime'], inplace=True, axis=1)
    residuals = []
    predictions = pred.reshape(pred.shape[0], pred.shape[1])
#   d = data.drop(['sessionUID', 'lap_time_remaining'], axis=1, inplace=True)
    names = data.columns
    err = testY-predictions
    for i in range(len(predictions)):
        test_X = pd.DataFrame(testX[i], columns = names)
        test_X['predictions'] = predictions[i]
        test_X['residuals'] = err[i]
        residuals.append(pd.DataFrame(test_X))
    final = pd.concat(residuals)
    print(final.info())
    x = final['currentLapTime']
    y = final['residuals']
    plt.scatter(x,y, s=0.05, c=y)
    plt.show()
    return final

if __name__ == '__main__':

    data = training_data()

    # data = pd.read_csv('../../Bayesian_RNN/sample_data.csv')
    # data.reset_index(inplace=True, drop=True)
    # data = pad_laps(data)

    # training_sessions = data.pop('sessionUID').to_numpy()
    #
    # data, scalers = scale_data(data)
    #
    # data['sessionUID'] = training_sessions

    trainX, testX, trainY, testY = format_data(data)

    # train_generator, test_generator, testX, testY, test_sessions = generate_data(data, labels)

    model = build_model(trainX)

    model, history = train_model(trainX, trainY, model)

    # model = tf.keras.models.load_model('lstm_normal.h5')

    predictions = model.predict(testX)

    test_dataframe = plot_residuals(testX, testY, predictions, data)
