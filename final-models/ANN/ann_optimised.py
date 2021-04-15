import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import pickle
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels as sm
import os
import sqlite3
import gc
import xgboost
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
            cur.execute('SELECT * FROM walk_forward')
            df = pd.DataFrame(cur.fetchall())
            data.append(df)

    names = list(map(lambda x: x[0], cur.description))
    df = pd.concat(data)
    df.columns = names
    df = df.drop(['frameIdentifier','bestLapTime', 'pkt_id', 'packetId', 'SessionTime', 'lap_time_remaining'], axis=1)
    df.set_index('index', inplace=True)

    return df


def series_to_supervised(data,n_in=1, n_out=1, dropnan=True):
    n_vars = len(data.columns)
    df = data
    cols= [data]
    names = data.columns
    for i in range(n_in, 0, -1):
        neg_shift = df.shift(i)
        minus_names = [f'{name}_t-{i}' for name in names]
        neg_shift.columns = minus_names
        cols.append(neg_shift)

    if n_out > 1:
        for i in range(0, n_out):
            pos_shift = df.shift(-i)
            plus_names = [f'{name}_t+{i}' for name in names]
            pos_shift.columns = plus_names
            cols.append(pos_shift)

    agg = pd.concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def build_model(trainX):
    print(f'Number of input dims: {trainX.shape[1]}')
    model = tf.keras.Sequential()
    model.add(Dense(16, input_dim=trainX.shape[1],  activation='sigmoid', kernel_initializer='normal'))
    model.add(Dense(64, activation='relu', kernel_initializer='normal'))
    model.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(16, activation='sigmoid', kernel_initializer='normal'))
    model.add(Dense(1))

    adam = tf.keras.optimizers.Adam(lr=10**-2)

    model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    print(model.summary())

    return model

def train_model(trainX, trainY, model):
    epochs=100
    callback = [EarlyStopping(monitor="loss", patience = 10, mode = 'auto', restore_best_weights=True),
                ModelCheckpoint('ann_optimised_model.h5')]
    history = model.fit(trainX, trainY, callbacks=callback, shuffle=False, epochs=epochs, batch_size=10)
    return model, history


def normalise_data(df):
    scalers = {}
    for col in list(df.columns):
        s = StandardScaler()
        df[col] = s.fit_transform(df[col].values.reshape(-1,1))
        scalers[col] = s
    print('Data Normalised')
    return df, scalers

def descale(df, scalers):
    for col in list(df.columns):
        df[col] = scalers[col].inverse_transform(df[col].values.reshape(-1,1))
    return df

def predictions(model, testX, testY, scalers):
    pred = model.predict(testX)
    testX = descale(testX, scalers)
    testX['finalLapTime'] = testY
    testX['predictions'] = pred.ravel()
    testX['residuals'] = testX['predictions'] - testX['finalLapTime']
    testX.rename({'finalLapTime':'truth'}, inplace=True)
    plt.scatter(testX['currentLapTime'], testX['residuals'], s=0.1)
    plt.show()
    return testX


if __name__ == '__main__':

    data = training_data()

    data.reset_index(drop=True, inplace=True)

    # data = pd.read_csv('../Bayesian_RNN/sample_data.csv')
    # data.drop(['index'], axis=1, inplace=True)

    df = series_to_supervised(data, n_in = 1, n_out=1, dropnan=True)

    df.drop(['finalLapTime_t-1', 'sessionUID_t-1'], axis=1, inplace=True)

    sessionUID = df.pop('sessionUID')

    labels = df.pop('finalLapTime')

    df, scalers = normalise_data(df)

    df['sessionUID'] = sessionUID


    trainX, testX, trainY, testY = train_test_split(df, labels, shuffle=True, test_size=0.2)



    trainX.drop(['sessionUID'], inplace=True, axis=1)


    testX = testX.sort_index()
    testY = testY.sort_index()

    test_sessions = testX.pop('sessionUID')

    print(trainX.info())
    print(testX.info())

    model = build_model(trainX)

    model, history = train_model(trainX, trainY, model)

    # model = tf.keras.models.load_model('ann.h5')

    pred = predictions(model, testX, testY, scalers)

    pred['sessionUID'] = test_sessions.to_numpy()

    pred.to_csv('ANN_optimised_predictions.csv')

    model.save('ann_optimised_model.h5')
