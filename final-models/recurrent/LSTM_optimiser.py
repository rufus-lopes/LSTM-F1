import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sqlite3
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)



def data():
    df = pd.read_csv('optimiser_data_lstm.csv')
    target = df.pop('finalLapTime')
    trainX, testX, trainY, testY = train_test_split(df, target, test_size=0.2, random_state=42, shuffle = False)
    trainX = trainX.to_numpy()
    testX = testX.to_numpy()
    trainY = trainY.to_numpy()
    testY = testY.to_numpy()
    return trainX, testX, trainY, testY

def model(trainX, testX, trainY, testY):
    X = trainX[0]
    model = Sequential()
    model.add(LSTM(units={{choice([8, 16, 32, 64, 128])}}, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(LeakyReLU(alpha={{uniform(0,1)}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(LSTM(units={{choice([8, 16, 32, 64, 128])}}, return_sequences=True))
    model.add(LeakyReLU(alpha={{uniform(0,1)}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(LSTM(units={{choice([8, 16, 32, 64, 128])}}, return_sequences=True))
    model.add(LeakyReLU(alpha={{uniform(0,1)}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1))
    adam = tf.keras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
    rmsprop = tf.keras.optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})
    sgd = tf.keras.optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})
    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    else:
        optim = sgd
    model.compile(optimizer=optim, loss='mse', metrics=['mae'])
    epochs = 100
    callback = [EarlyStopping(monitor="loss", min_delta = 0.0001, patience = 3, mode = 'auto', restore_best_weights=True), ModelCheckpoint('generator_lstm.h5')]
    model.fit(x=trainX, y=trainY, batch_size=1, epochs=epochs, shuffle=False, callbacks=callback)
    score, acc = model.evaluate(testX, testY)
    print(score)
    return {'loss': acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':


    trainX, testX, trainY, testY = data()
    best_run, best_model = optim.minimize(model = model,
                                        data = data,
                                        algo = tpe.suggest,
                                        max_evals = 5,
                                        trials = Trials(),
                                        )


    print('best model parameters:')
    print(best_run)

    print('best model score: ')
    print(best_model.evaluate(testX, testY))

    best_run.save('best_run.h5')
