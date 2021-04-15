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


def predictions(model, testX, testY):
    testX['predictions'] = model.predict(testX)
    testX['truth'] = testY
    testX['residuals'] = testX['predictions'] - testX['truth']
    plt.scatter(testX['currentLapTime'], testX['residuals'])
    plt.show()
    return testX

def xg_boost(trainX, trainY):
    model = xgboost.XGBRegressor(colsample_bytree = 0.8, gamma=0.03, learning_rate = 0.1, max_depth = 5,
                                   min_child_weight = 1.5, n_estimators = 10000, subsample=0.95,
                                   tree_method = 'gpu_hist')
    model.fit(trainX, trainY)
    print('XGBoost fit')
    return model

if __name__ == '__main__':

    data = training_data()

    data.reset_index(inplace=True, drop=True)

    label = data.pop('finalLapTime')

    trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.2, shuffle=True)

    testX.sort_index(inplace=True)
    testY.sort_index(inplace=True)

    trainX.drop(['sessionUID'], inplace=True, axis=1)

    test_sessionUID = testX.pop('sessionUID').to_numpy()

    model = xg_boost(trainX, trainY)

    pred = predictions(model, testX, testY)

    pred['sessionUID'] = test_sessionUID

    pred.to_csv('prediction_rolling_csv/XGBoost_rolling_predictions.csv')

    file = 'saved_models/XGBoost_rolling_model.sav'

    pickle.dump(model, open(file, 'wb'))
