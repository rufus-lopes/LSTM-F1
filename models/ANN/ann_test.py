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
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

def training_data():

    dir = '../../SQL_Data/constant_setup'
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
    df = df.drop(['frameIdentifier','bestLapTime', 'pkt_id', 'packetId', 'SessionTime', 'finalLapTime'], axis=1)
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

    model = tf.keras.Sequential()
    model.add(Dense(128, input_shape=(trainX.shape[1],), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))

    adam = tf.keras.optimizers.Adam(lr=0.1)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    print(model.summary())

    return model

def train_model(trainX, trainY, model):
    epochs=2
    callback = [EarlyStopping(monitor="loss", min_delta = 0.0001, patience = 10, mode = 'auto', restore_best_weights=True)]
    history = model.fit(trainX, trainY, callbacks=callback, shuffle=False, epochs=epochs, batch_size=1)
    return model, history

def make_predictions(testX, testY, model):
    test_df = pd.DataFrame(testX)
    print(test_df.head())
    predictions = pd.DataFrame()
    truth = pd.DataFrame(testY)
    truth['sessionUID'] = testX['sessionUID']
    truth['currentLapNum'] = test_df['currentLapNum']
    session_groups = test_df.groupby('sessionUID')
    truth_session_groups = truth.groupby('sessionUID')
    lap_num = 1
    for s in list(session_groups.groups):
        session = session_groups.get_group(s)
        truth_session = truth_session_groups.get_group(s)
        lap_groups = session.groupby('currentLapNum')
        truth_lap_groups = truth_session.groupby('currentLapNum')
        for l in list(lap_groups.groups):
            lap = lap_groups.get_group(l)
            truth_lap = truth_lap_groups.get_group(l)
            truth_lap.drop(['currentLapNum', 'sessionUID'], axis=1, inplace=True)
            truth_lap.reset_index(drop=True, inplace=True)
            lap_truth = truth_lap.to_numpy()
            print(lap_truth.shape)
            lap_truth = lap_truth.ravel()
            lap.drop(['sessionUID'], axis=1, inplace=True)
            pred = model.predict(lap).ravel()
            predictions[f'lap_{lap_num}_predictions'] = pd.Series(pred)
            predictions[f'lap_{lap_num}_truth'] = pd.Series(lap_truth)
            lap_num+=1


    print(predictions.info())
    predictions.to_csv('ann_predictions.csv')
    predictions.plot()
    plt.show()



if __name__ == '__main__':

    data = training_data()

    # data = pd.read_csv('../Bayesian_RNN/sample_data.csv')
    # data.drop(['index'], axis=1, inplace=True)
    # print(data.info())

    df = series_to_supervised(data, n_in = 1, n_out=1, dropnan=True)

    labels = df.pop('lap_time_remaining')

    df.drop('sessionUID_t-1', axis=1, inplace=True)

    names = df.columns

    trainX, testX, trainY, testY = train_test_split(df, labels, shuffle=False, test_size=0.2)

    trainX.drop(['sessionUID'], inplace=True, axis=1)

    model = build_model(trainX)

    model, history = train_model(trainX, trainY, model)

    make_predictions(testX, testY, model)
