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

def sub_sample(df):
    arr = []
    session_groups = df.groupby('sessionUID')
    for s in list(session_groups.groups):
        session = session_groups.get_group(s)
        lap_groups = session.groupby('currentLapNum')
        for l in list(lap_groups.groups):
            lap = lap_groups.get_group(l)
            df2 = lap[lap.index % 10 == 0]  # Selects every 10th row starting from 0
            arr.append(df2)

    sub_sampled_data = pd.concat(arr)
    print(f'Full sub sample shape {sub_sampled_data.shape}')
    return sub_sampled_data

def generate_data(training, target):
    trainX, testX, trainY, testY = train_test_split(training, target, test_size=0.2, random_state=42, shuffle = False)


    test_sessions = testX.pop('sessionUID').to_numpy()
    train_session = trainX.pop('sessionUID')

    test_df_names = list(testX.columns)


    trainX = trainX.to_numpy()
    testX = testX.to_numpy()
    trainY = trainY.to_numpy()
    testY = testY.to_numpy()
    stride = 5
    look_back = 20
    batch_size = 2
    train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(trainX, trainY, length=look_back, sampling_rate=1, stride=1, batch_size=batch_size)
    test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(testX, testY, length=look_back, sampling_rate=1, stride=1, batch_size=batch_size)

    return train_generator, test_generator, testX, testY, test_sessions

def build_model(train_generator):

    X, y = train_generator[0]
    learning_rate = 0.0001
    units = 128
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(X.shape[1], X.shape[2]),    kernel_initializer="random_normal",))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units, return_sequences=True)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
    adam = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    print(model.summary())
    return model

def train_model(train_generator, model):
    callback = [EarlyStopping(monitor="loss", min_delta = 0.0001, patience = 3, mode = 'auto', restore_best_weights=True),
                ModelCheckpoint('generator_sample_lstm.h5')]
    epochs = 100
    history = model.fit(train_generator,
                        callbacks=callback,
                        shuffle=False,
                        epochs=epochs)
    return model, history

def predictions(model, test_generator, testX, testY, scalers, names, test_sessions):

    pred = model.predict(test_generator)
    predictions = scalers['scaler_finalLapTime'].inverse_transform(pred)
    truth = [y for (x,y) in test_generator]
    truth = np.array(truth)
    truth = truth.ravel()

    print(predictions.shape)
    print(truth.shape)
    
    truth = scalers['scaler_finalLapTime'].inverse_transform(truth.reshape(-1,1))
    err = truth-predictions
    ans = pd.DataFrame()
    ans['truth'] = truth.ravel()
    ans['predictions'] = predictions.ravel()
    ans['residuals'] = err.ravel()
    test_df = pd.DataFrame(testX)
    names = names
    names = names[:-1]
    test_df.columns = names
    for col in list(test_df.columns):
        us = scalers[f'scaler_{col}'].inverse_transform(test_df[col].values.reshape(-1,1))
        us = np.reshape(us, len(us))
        test_df[col] = us

    test_df['sessionUID'] = test_sessions
    test_df['truth'] = pd.Series(ans['truth'])
    test_df['predictions'] = pd.Series(ans['predictions'])
    test_df['residuals'] = pd.Series(ans['residuals'])
    test_df.to_csv('LSTM_sampled_predictions.csv')

    plt.scatter(x = test_df['currentLapTime'], y = test_df['residuals'], s=1)
    plt.show()

    return test_df



if __name__ == '__main__':

    data = training_data()

    data = sub_sample(data)

    data.reset_index(inplace=True, drop=True)

    data = pad_laps(data)

    training_sessions = data.pop('sessionUID').to_numpy()

    data, scalers = scale_data(data)

    data['sessionUID'] = training_sessions

    labels = data.pop('finalLapTime')

    names = list(data.columns)

    train_generator, test_generator, testX, testY, test_sessions = generate_data(data, labels)

    model = build_model(train_generator)

    # model, hitory = train_model(train_generator, model)

    model = tf.keras.models.load_model('generator_sample_lstm.h5')

    pred = predictions(model, test_generator, testX, testY, scalers, names, test_sessions)
