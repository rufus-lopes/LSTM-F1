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
from statsmodels.stats.stattools import durbin_watson
import pickle


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

def importance(model, trainX):
    coefs = model.coef_.tolist()
    coefs = [abs(x) for x in coefs]
    names = list(trainX.columns)
    zipper = zip(names, coefs)
    imp = dict(zipper)
    imp = {k: v for k, v in sorted(imp.items(), key=lambda item: item[1])}

    x, y = [], []
    for i in imp.keys():
        x.append(i)
        y.append(imp[i])
    x = x[len(x)-10:]
    y = y[len(y)-10:]
    plt.figure(figsize=(15,8))
    plt.bar([p for p in range(len(x))], y, tick_label=x)
    plt.xticks(rotation=30, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel('Feature Coefficient', fontsize=20, fontweight='bold')
    plt.show()

def linear_regression(trainX, trainY):
    model = LinearRegression()
    model.fit(trainX, trainY)
    print('Linear Regression fit' )
    importance(model, trainX)
    return model

def predictions(model, testX, testY):
    testX['predictions'] = model.predict(testX)
    testX['truth'] = testY
    testX['residuals'] = testX['predictions'] - testX['truth']
    plt.scatter(testX['currentLapTime'], testX['residuals'], s=0.1)
    plt.show()
    return testX

def normalise_data(df):
    scalers = {}
    for col in list(df.columns):
        s = StandardScaler()
        df[col] = s.fit_transform(df[col].values.reshape(-1,1))
        scalers[col] = s
    print('Data Normalised')
    return df, scalers

if __name__ == '__main__':

    data = training_data()

    sessions = data.pop('sessionUID').to_numpy()

    data, scalers = normalise_data(data)

    data['sessionUID'] = sessions

    data.reset_index(inplace=True, drop=True)

    data = series_to_supervised(data)

    label = data.pop('finalLapTime')

    data.drop(['finalLapTime_t-1', 'sessionUID_t-1'], axis=1, inplace=True)

    trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.2, shuffle=True)

    testX.sort_index(inplace=True)
    testY.sort_index(inplace=True)

    trainX.drop('sessionUID', inplace=True, axis=1)

    linear_model = linear_regression(trainX,trainY)

    test_sessions = testX.pop('sessionUID').to_numpy()

    pred = predictions(linear_model, testX, testY)

    pred['sessionUID'] = test_sessions

    pred.to_csv('prediction_csv/Linear_walk_predictions.csv')

    file = 'saved_models/linear_model.sav'
    pickle.dump(linear_model, open(file, 'wb'))
