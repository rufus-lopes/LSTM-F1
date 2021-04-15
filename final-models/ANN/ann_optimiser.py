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
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform



def get_data():
    df = pd.read_csv('optimiser_data.csv')
    labels = df.pop('finalLapTime')
    trainX, testX, trainY, testY = train_test_split(df, labels, shuffle=True, test_size=0.2)
    testX = testX.sort_index()
    testY = testY.sort_index()
    return trainX, testX, trainY, testY


def model(trainX, testX, testY, trainY):
    model = tf.keras.Sequential()
    model.add(Dense(units={{choice([8, 16, 32, 64, 128])}}, input_dim=trainX.shape[1],  activation={{choice(['relu', 'sigmoid'])}}, kernel_initializer={{choice(['normal', 'random_normal'])}}))
    model.add(Dense(units={{choice([8, 16, 32, 64, 128])}}, activation={{choice(['relu', 'sigmoid'])}}, kernel_initializer={{choice(['normal', 'random_normal'])}}))
    model.add(Dense(units={{choice([8, 16, 32, 64, 128])}}, activation={{choice(['relu', 'sigmoid'])}}, kernel_initializer={{choice(['normal', 'random_normal'])}}))
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(units={{choice([8, 16, 32, 64, 128])}}, activation={{choice(['relu', 'sigmoid'])}}, kernel_initializer={{choice(['normal', 'random_normal'])}}))
    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
    rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})
    sgd = keras.optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})
    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    else:
        optim = sgd
    model.compile(optimizer=optim, loss='mse', metrics=['mae'])
    epochs=100
    callback = [EarlyStopping(monitor="loss", patience = 10, mode = 'auto', restore_best_weights=True), ModelCheckpoint('ann.h5')]
    history = model.fit(trainX, trainY, callbacks=callback, shuffle=False, epochs=epochs, batch_size=10)
    score, acc = model.evaluate(testX, testY)
    return {'loss': acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':



    trainX, testX, trainY, testY = get_data()


    best_run, best_model = optim.minimize(model = model,
                                        data = get_data,
                                        algo = tpe.suggest,
                                        max_evals = 5,
                                        trials = Trials(),)

    print('best model parameters:')
    print(best_run)

    print('best model score: ')
    print(best_model.evaluate(testX, testY))
