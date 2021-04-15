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
from xgboost import plot_importance

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


def predictions(model, testX, testY):
    testX['predictions'] = model.predict(testX)
    testX['truth'] = testY
    testX['residuals'] = testX['predictions'] - testX['truth']
    plt.scatter(testX['currentLapTime'], testX['residuals'])
    plt.show()
    return testX

def xg_boost(trainX, trainY):
    model = xgboost.XGBRegressor(tree_method = 'gpu_hist')
    model.fit(trainX, trainY)
    print('XGBoost fit')
    ax = plot_importance(model, max_num_features=15, height=0.8)
    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(12)
                tick.label.set_fontweight('bold')

    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(13)
                tick.label.set_rotation(45)
                tick.label.set_fontweight('bold')


    ax.xaxis.set_label_text('Feature Importance Score', fontsize=25, fontweight='bold')

    # ax.yaxis.set_label_text('Feature', fontsize=25, fontweight='bold')
    # ax.margins(tight=True)
    plt.savefig('feature_importance_xgb.png')
    plt.show()
    return model

if __name__ == '__main__':

    data = training_data()

    data.reset_index(inplace=True, drop=True)

    # data = series_to_supervised(data)

    label = data.pop('finalLapTime')

    # data.drop(['finalLapTime_t-1', 'sessionUID_t-1'], axis=1, inplace=True)

    trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.2, shuffle=True)

    print(trainX.info())

    testX.sort_index(inplace=True)
    testY.sort_index(inplace=True)

    trainX.drop(['sessionUID'], inplace=True, axis=1)

    test_sessionUID = testX.pop('sessionUID').to_numpy()

    model = xg_boost(trainX, trainY)

    pred = predictions(model, testX, testY)

    pred['sessionUID'] = test_sessionUID

    pred.to_csv('prediction_csv/XGBoost_naive_predictions.csv')

    file = 'saved_models/XGBoost_naive_model.sav'

    pickle.dump(model, open(file, 'wb'))
