import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
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
from sklearn.svm import SVR


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


def show_predictions_svr(model, df):
    predictions = pd.DataFrame()
    session_groups = df.groupby('sessionUID')
    lap_num = 1
    for s in list(session_groups.groups):
        session = session_groups.get_group(s)
        lap_groups = session.groupby('currentLapNum')
        for l in list(lap_groups.groups):
            lap = lap_groups.get_group(l)
            lap.drop(['sessionUID'], axis=1, inplace=True)
            pred = model.predict(lap)
            predictions[f'lap_{lap_num}'] = pd.Series(pred)
            lap_num+=1
    predictions.plot()
    plt.savefig('SVR_predictions.svg')
    predictions.to_csv('SVR_predictions.csv')
    return predictions

def svr_model(trainX, trainY):
    model = SVR()
    model.fit(trainX, trainY)
    print('SVR Fit')
    return model


if __name__ == '__main__':

    # data = training_data()
    data = pd.read_csv('../Bayesian_RNN/sample_data.csv')
    data.drop(['index'], axis=1, inplace=True)

    print(data.info())

    df = series_to_supervised(data, n_in = 1, n_out=1, dropnan=True)

    df.drop('lap_time_remaining_t-1', axis=1, inplace=True)

    label = df.pop('lap_time_remaining')

    sessionUID = df.pop('sessionUID')
    df.drop('sessionUID_t-1', axis=1, inplace=True)



    trainX, testX, trainY, testY = train_test_split(df, label, shuffle=False, test_size=0.2)

    prediction_data = df
    prediction_data['sessionUID'] = sessionUID

    SVR_model = svr_model(trainX, trainY)

    SVR_predictions = show_predictions_svr(SVR_model,prediction_data)

    svr_pred = SVR_model.predict(testX)

    print(f'SVR Mean Average Error is {mae(testY, svr_pred)}')
