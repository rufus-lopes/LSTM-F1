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

def format_data(data):
    '''

    seperates data first by session, then by lap, before padding each array so that
    they are all the same length for model input.
    Performs test train split also

    '''
    data.reset_index(drop=True, inplace=True)

    session_groups = data.groupby('sessionUID')
    training_data = []
    target_data = []
    total_laps = 0
    for s in list(session_groups.groups):
        session = session_groups.get_group(s)
        lap_groups = session.groupby('currentLapNum')
        total_laps += len(lap_groups)
        for l in list(lap_groups.groups):
            lap = lap_groups.get_group(l)
            lap = lap.drop(['sessionUID', 'sessionUID_t-1'], axis=1)
            target_data.append(lap.pop('lap_time_remaining'))
            training_data.append(lap)

    training = [x.to_numpy() for x in training_data]
    target = [y.to_numpy() for y in target_data]
    print(f'Total Laps: {total_laps}')

    max_timesteps = 10000 # max(training, key=len).shape[0]
    num_rows_to_add = [max_timesteps-l.shape[0] for l in training]

    training_pad = []
    target_pad = []
    print(f'max timesteps : {max_timesteps}')

    for i in range(len(training)):
        rows_to_add = num_rows_to_add[i]

        training_arr = training[i]
        training_append = np.zeros((rows_to_add, training[0].shape[1]), dtype=float)
        training_array = np.vstack((training_arr, training_append))
        training_pad.append(training_array)

        target_arr = target[i]
        target_append = np.zeros((rows_to_add), dtype=float)
        target_array = np.concatenate([target_arr, np.zeros(rows_to_add)])
        target_pad.append(target_array)

    split = int(total_laps*0.9)

    X_train = training_pad[:split]
    X_test = training_pad[split:]
    y_train = target_pad[:split]
    y_test = target_pad[split:]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_train = np.array(y_train)

    print('Data Formatted')

    return X_train, X_test, y_train, y_test

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

def show_predictions_linear_reg(model, df):
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
    plt.savefig('Linear_Regression_predictions.svg')
    predictions.to_csv('Linear_Regression_predictions.csv')
    return predictions

def show_predictions_xgb(model, df):
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
    plt.savefig('XGboost_predictions.svg')
    predictions.to_csv('XGboost_predictions.csv')
    return predictions

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

def linear_regression(trainX, trainY):
    model = LinearRegression()
    model.fit(trainX, trainY)
    print('Linear Regression fit' )
    return model

def xg_boost(trainX, trainY):
    model = xgboost.XGBRegressor(colsample_bytree = 0.8, gamma=0.03, learning_rate = 0.1, max_depth = 5, min_child_weight = 1.5, n_estimators = 10000, subsample=0.95, tree_method = 'gpu_hist')
    model.fit(trainX, trainY)
    print('XGBoost fit')
    return model

def svr_model(trainX, trainY):
    model = SVR()
    model.fit(trainX, trainY)
    print('SVR Fit')
    return model

if __name__ == '__main__':

    data = training_data()
    # data = pd.read_csv('../Bayesian_RNN/sample_data.csv')
    # data.drop(['index'], axis=1, inplace=True)

    df = series_to_supervised(data, n_in = 1, n_out=1, dropnan=True)

    df.drop('lap_time_remaining_t-1', axis=1, inplace=True)

    label = df.pop('lap_time_remaining')

    sessionUID = df.pop('sessionUID')
    df.drop('sessionUID_t-1', axis=1, inplace=True)

    trainX, testX, trainY, testY = train_test_split(df, label, shuffle=False, test_size=0.2)

    linear_model = linear_regression(trainX,trainY)

    xgb = xg_boost(trainX, trainY)

    # SVR_model = svr_model(trainX, trainY)

    linear_pred = linear_model.predict(testX)

    xgb_pred =xgb.predict(testX)

    # svr_pred = SVR_model.predict(testX)


    prediction_data = df
    prediction_data['sessionUID'] = sessionUID


    linear_predictions = show_predictions_linear_reg(linear_model, prediction_data)

    xgb_predictions = show_predictions_xgb(xgb, prediction_data)

    # SVR_predictions = show_predictions_svr()


    print(f'LinearRegression Mean Average Error is {mae(testY, linear_pred)}')

    print(f'XGBoost Mean Average Error is {mae(testY, xgb_pred)}')

    # print(f'SVR Mean Average Error is {mae(testY, svr_pred)}')
