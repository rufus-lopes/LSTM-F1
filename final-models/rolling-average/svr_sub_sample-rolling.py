import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels as sm
import os
import sqlite3
import gc
import xgboost
from sklearn.svm import SVR
import pickle
from sklearn.pipeline import Pipeline

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


def svr_model(trainX, trainY):
    steps = [('sca', MinMaxScaler(feature_range=(-1,1))), ('m', SVR(cache_size=1000))]
    model = Pipeline(steps=steps)
    model.fit(trainX, trainY)
    print('SVR Fit')
    return model

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

def scale_data(trainX, testX):
    trainX_scaler = MinMaxScaler(feature_range=(-1,1))
    testX_scaler = MinMaxScaler(feature_range=(-1,1))
    trainX = pd.DataFrame(trainX_scaler.fit_transform(trainX), columns = list(trainX.columns))
    testX = pd.DataFrame(testX_scaler.fit_transform(testX), columns = list(testX.columns))
    return trainX, testX, trainX_scaler, testX_scaler

def predictions(model, testX, testY):
    testX['predictions'] = model.predict(testX)
    testX['truth'] = testY
    testX['residuals'] = testX['predictions'] - testX['truth']
    plt.scatter(testX['currentLapTime'], testX['residuals'])
    plt.show()
    return testX


if __name__ == '__main__':

    data = training_data()

    data.reset_index(inplace=True, drop=True)


    data = sub_sample(data)

    label = data.pop('finalLapTime')

    trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.2, shuffle=True)

    testX.sort_index(inplace=True)
    testY.sort_index(inplace=True)

    trainX.drop(['sessionUID'], inplace=True, axis=1)

    testX_sessions = testX.pop('sessionUID').to_numpy()

    SVR_model = svr_model(trainX, trainY)

    pred = predictions(SVR_model, testX, testY)

    pred['sessionUID'] = testX_sessions

    pred.to_csv('prediction_rolling_csv/SVR_sample_rolling_predictions.csv')

    file = 'saved_models/svr_sample_rolling_model.sav'
    pickle.dump(SVR_model, open(file, 'wb'))
