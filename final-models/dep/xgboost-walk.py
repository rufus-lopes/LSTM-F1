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

def show_predictions_xgb(model, testX, testY):
    predictions =[]
    session_groups = testX.groupby('sessionUID')
    test_session_groups = testY.groupby('sessionUID')
    for s in list(session_groups.groups):
        session = session_groups.get_group(s)
        test_session = test_session_groups.get_group(s)
        test_lap_groups = test_session.groupby('currentLapNum')
        lap_groups = session.groupby('currentLapNum')
        for l in list(lap_groups.groups):
            test_lap = test_lap_groups.get_group(l)
            test_lap.drop(['currentLapNum', 'sessionUID'], axis=1, inplace=True)
            lap = lap_groups.get_group(l)
            sessionUID = lap.pop('sessionUID')
            lap['predictions'] = model.predict(lap)
            lap['sessionUID'] = sessionUID
            lap['truth'] = test_lap
            predictions.append(lap)
    predictions = pd.concat(predictions)
    x = predictions['currentLapTime']
    y = predictions['predictions']
    plt.scatter(x=x, y=y, s=0.05)
    # plt.savefig('XGboost_predictions.svg')
    plt.show()
    predictions.to_csv('prediction_csv/XGBoost_predictions.csv')
    return predictions


def xg_boost(trainX, trainY):
    # model = xgboost.XGBRegressor(colsample_bytree = 0.8, gamma=0.03, learning_rate = 0.1, max_depth = 5, min_child_weight = 1.5, n_estimators = 10000, subsample=0.95, tree_method = 'gpu_hist')
    model = xgboost.XGBRegressor(tree_method='gpu_hist')
    model.fit(trainX, trainY)
    print('XGBoost fit')
    return model

if __name__ == '__main__':

    data = training_data()
    # data = pd.read_csv('../Bayesian_RNN/sample_data.csv')
    # data.drop(['index'], axis=1, inplace=True)

    # df = series_to_supervised(data, n_in = 1, n_out=1, dropnan=True)

    # df.drop('finalLapTime_t-1', axis=1, inplace=True)
    df = data
    label = pd.DataFrame(df.pop('finalLapTime'))

    # df.drop('sessionUID_t-1', axis=1, inplace=True)

    trainX, testX, trainY, testY = train_test_split(df, label, shuffle=False, test_size=0.2)

    train_sessionUID = trainX.pop('sessionUID')

    xgb = xg_boost(trainX, trainY)

    testY['sessionUID'] = testX['sessionUID']
    testY['currentLapNum'] = testX['currentLapNum']

    print(testY.info())

    xgb_predictions = show_predictions_xgb(xgb, testX, testY)

    testX.drop('sessionUID', axis=1, inplace=True)

    file = 'saved_models/XGBoost_model.sav'
    pickle.dump(xgb, open(file, 'wb'))

    xgb_pred =xgb.predict(testX)

    truth = testY['finalLapTime']

    print(f'XGBoost Mean Average Error is {mae(truth, xgb_pred)}')
