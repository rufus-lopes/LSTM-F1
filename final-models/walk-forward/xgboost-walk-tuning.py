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
from sklearn.model_selection import RandomizedSearchCV
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

def search(trainX, trainY):

    parameters_for_testing = {
   'colsample_bytree':[0.4,0.6,0.8],
   'gamma':[0,0.03,0.1,0.3],
   'min_child_weight':[1.5,6,10],
   'learning_rate':[0.1,0.07],
   'max_depth':[3,5],
   'n_estimators':[500],
   'subsample':[0.6,0.95]
}

    parameters_for_testing = {'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [100],
}
    xgb_model = xgboost.XGBRegressor(tree_method='gpu_hist')

    gsearch1 = RandomizedSearchCV(estimator = xgb_model, param_distributions = parameters_for_testing, n_jobs=6, verbose=10, scoring='neg_mean_squared_error')

    gsearch1.fit(trainX, trainY)

    return gsearch1

def xg_boost(trainX, trainY, gsearch1):
    params = gsearch1.best_params_
    model = xgboost.XGBRegressor(tree_method = 'gpu_hist', params=params)
    model.fit(trainX, trainY)
    print('XGBoost fit')
    plot_importance(model, max_num_features=15)
    plt.show()
    return model

if __name__ == '__main__':

    data = training_data()

    data = sub_sample(data)

    data.reset_index(inplace=True, drop=True)

    data = series_to_supervised(data)

    label = data.pop('finalLapTime')

    data.drop(['finalLapTime_t-1', 'sessionUID_t-1'], axis=1, inplace=True)

    trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.2, shuffle=True)

    print(trainX.info())

    testX.sort_index(inplace=True)
    testY.sort_index(inplace=True)

    trainX.drop(['sessionUID'], inplace=True, axis=1)

    test_sessionUID = testX.pop('sessionUID').to_numpy()

    gsearch = search(trainX, trainY)

    model = xg_boost(trainX, trainY, gsearch)

    pred = predictions(model, testX, testY)

    pred['sessionUID'] = test_sessionUID


    print('Best parameters for XGBoost: ')
    print(gsearch.best_params_)

    pred.to_csv('prediction_csv/XGBoost_tuned_predictions.csv')

    file = 'saved_models/XGBoost_tuned_model.sav'

    pickle.dump(model, open(file, 'wb'))
