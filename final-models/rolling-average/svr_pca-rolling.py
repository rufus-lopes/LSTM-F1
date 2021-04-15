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
from sklearn.decomposition import PCA
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
    model = SVR(cache_size=1000)
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
            df2 = lap[lap.index % 10 == 0]  # Selects every 5th row starting from 0
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


def get_models():
	models = dict()
	for i in range(10,20):
		steps = [('scaler', StandardScaler()),('pca', PCA(n_components=i)), ('m', SVR())]
		models[str(i)] = Pipeline(steps=steps)
	return models

def evaluate_model(model, trainX, trainY, testX, testY):
    svr = model.fit(trainX, trainY)
    score = model.score(testX, testY)
    return score


if __name__ == '__main__':

    data = training_data()

    label = pd.DataFrame(data.pop('finalLapTime'))


    # df = series_to_supervised(data, n_in = 1, n_out=1, dropnan=True)

    # sessionUID = df.pop('sessionUID')

    trainX, testX, trainY, testY = train_test_split(data, label, shuffle=False, test_size=0.2)

    trainX_sessionUID = trainX.pop('sessionUID').to_numpy()
    testX_sessionUID = testX.pop('sessionUID').to_numpy()

    # trainX, testX, trainX_scaler, testX_scaler = scale_data(trainX, testX)

    testX_laps = testX['currentLapNum'].to_numpy()
    testY['currentLapNum'] = testX_laps
    testY['sessionUID'] = testX_sessionUID

    #train model here

    models = get_models()
    results, names = list(), list()
    for name, model in models.items():
    	scores = evaluate_model(model, trainX, trainY, testX, testY)
    	results.append(scores)
    	names.append(name)
    	print(f'Model {name}:  {scores}')

    plt.boxplot(results, labels=names, showmeans=True)
    plt.xticks(rotation=45)
    plt.savefig('best_pca.png')
    plt.show()


############################


    # testX['sessionUID'] = testX_sessionUID
    #
    # SVR_predictions = show_predictions_svr(SVR_model,testX, testY, testX_scaler)
    # testX.drop(['sessionUID'], axis=1, inplace=True)
    # svr_pred = SVR_model.predict(testX)
    #
    # testY.drop(['sessionUID', 'currentLapNum'], axis=1, inplace=True)
    # print(f'SVR Mean Average Error is {mae(testY, svr_pred)}')
