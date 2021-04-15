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
            cur.execute('SELECT * FROM rolling_average')
            df = pd.DataFrame(cur.fetchall())
            data.append(df)

    names = list(map(lambda x: x[0], cur.description))
    df = pd.concat(data)
    df.columns = names
    df = df.drop(['frameIdentifier','bestLapTime', 'pkt_id', 'packetId', 'SessionTime',], axis=1)
    df.set_index('index', inplace=True)

    return df


def build_model(trainX):
    print(f'Number of input dims: {trainX.shape[1]}')
    model = tf.keras.Sequential()
    model.add(Dense(64, input_dim=trainX.shape[1]))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.2))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    adam = tf.keras.optimizers.Adam(lr=0.2)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    print(model.summary())

    return model

def train_model(trainX, trainY, model):
    epochs=150
    callback = [EarlyStopping(monitor="loss", patience = 10, mode = 'auto', restore_best_weights=True),
                ModelCheckpoint('ann.h5')]
    history = model.fit(trainX, trainY, callbacks=callback, shuffle=False, epochs=epochs, batch_size=10)
    return model, history

def make_predictions(testX, testY, model, scalers):
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
            print(lap.shape)
            lap.drop(['sessionUID'], axis=1, inplace=True)
            pred = model.predict(lap).ravel()
            pred = scalers['finalLapTime'].inverse_transform(pred.reshape(-1,1)).ravel()
            predictions[f'lap_{lap_num}_predictions'] = pd.Series(pred)
            # predictions[f'lap_{lap_num}_truth'] = pd.Series(lap_truth)
            lap_num+=1

    predictions.to_csv('ann_predictions.csv')
    predictions.plot()
    plt.show()




def normalise_data(df):
    scalers = {}
    for col in list(df.columns):
        s = MinMaxScaler(feature_range=(-1,1))
        df[col] = s.fit_transform(df[col].values.reshape(-1,1))
        scalers[col] = s
    return df, scalers



if __name__ == '__main__':

    data = training_data()

    # data = pd.read_csv('../Bayesian_RNN/sample_data.csv')
    # data.drop(['index'], axis=1, inplace=True)
    # print(data.info())

    sessionUID = data.pop('sessionUID')

    df, scalers = normalise_data(data)

    df['sessionUID'] = sessionUID

    df = series_to_supervised(data, n_in = 1, n_out=1, dropnan=True)


    labels = df.pop('finalLapTime')

    df.drop(['finalLapTime_t-1'], axis=1, inplace=True)

    df.drop('sessionUID_t-1', axis=1, inplace=True)

    print(df)

    names = df.columns

    trainX, testX, trainY, testY = train_test_split(df, labels, shuffle=False, test_size=0.2)

    trainX.drop(['sessionUID'], inplace=True, axis=1)

    # model = build_model(trainX)

    # model, history = train_model(trainX, trainY, model)

    model = tf.keras.models.load_model('ann.h5')

    # test_predictions(model, testX, testY)

    make_predictions(testX, testY, model, scalers)

    # pred = show_predictions(model, testX)

    model.save('saved_models/ann.h5')
