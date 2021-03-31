import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import sqlite3
import pandas as pd
from pathlib import Path
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices[0])
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def format_data(data):
    '''

    seperates data first by session, then by lap, before padding each array so that
    they are all the same length for model input.
    Performs test train split also

    '''
    data.reset_index(drop=True, inplace=True)

    scalers = {}
    sessionUIDs = data.pop('sessionUID')
    lap_number = data.pop('currentLapNum')
    for i in data.columns:
        scaler = MinMaxScaler(feature_range=(-1,1))
        s = scaler.fit_transform(data[i].values.reshape(-1,1))
        s = np.reshape(s, len(s))
        scalers['scaler_'+ i ] = scaler
        data[i] = s

    data['sessionUID'] = sessionUIDs
    data['currentLapNum'] = lap_number
    session_groups = data.groupby('sessionUID')
    training_data = []
    target_data = []
    total_laps = 0
    print(data.info())
    for s in list(session_groups.groups):
        session = session_groups.get_group(s)
        lap_groups = session.groupby('currentLapNum')
        total_laps += len(lap_groups)
        for l in list(lap_groups.groups):
            lap = lap_groups.get_group(l)
            lap = lap.drop(['sessionUID'], axis=1)
            target_data.append(lap.pop('lap_time_remaining'))
            training_data.append(lap)
    training = [x.to_numpy() for x in training_data]
    target = [y.to_numpy() for y in target_data]
    print(f'Total Laps: {total_laps}')

    max_timesteps = max(training, key=len).shape[0]
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

    return X_train, X_test, y_train, y_test, scalers, num_rows_to_add

def import_training_data():

    ''' imports training data as a pandas DataFrame '''

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
    df = df.drop(['frameIdentifier','bestLapTime', 'pkt_id', 'packetId', 'SessionTime'], axis=1)
    df.set_index('index', inplace=True)

    print('Data Imported')

    return df

def single_generator(trainX, testX, trainY, testY):
    ''' Creates train and test generators from data for a single lap/sequence'''

    timesteps = trainX.shape[1]-1
    look_back = 5
    batch_size = 1


    train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(trainX, trainY, length=look_back, sampling_rate=1, stride=1, batch_size=batch_size)
    test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(testX, testY, length=look_back, sampling_rate=1, stride=1, batch_size=1)

    return train_generator, test_generator

def multi_generator(trainX, testX, trainY, testY):

    ''' Creates a train and test generator for every single
        and stores them in an array '''


    timesteps = trainX[0].shape[1]-1
    look_back = 5
    batch_size = 1

    train_gens = []
    test_gens = []
    print(trainX.shape[0])
    for i in range(trainX.shape[0]):
        train_ = tf.keras.preprocessing.sequence.TimeseriesGenerator(trainX[i], trainY[i], length=look_back, sampling_rate=1, stride=1, batch_size=batch_size)
        train_gens.append(train_)


    for j in range(testY.shape[0]):
        test_ = tf.keras.preprocessing.sequence.TimeseriesGenerator(testX[j], testY[j], length=look_back, sampling_rate=1, stride=1, batch_size=1)
        test_gens.append(test_)

    return train_gens, test_gens

def print_setup(train_generator, test_generator):

    ''' Just prints out some setup info '''
    train_X, train_y = train_generator[0]
    test_X, test_y = test_generator[0]

    train_samples = train_X.shape[0]*len(train_generator)
    test_samples = test_X.shape[0]*len(test_generator)

    print("Total Records: {}".format(len(trainX)+len(testX)))
    print("Number of samples in training set (.9 * n): trainX = {}".format(trainX.shape[0]))
    print("Number of samples in testing set (.1 * n): testX = {}".format(testX.shape[0]))
    print("Number of total samples in training feature set: {}".format(train_samples))
    print("Number of samples in testing feature set: {}".format(test_samples))
    print("Number of features in training set : {}".format(train_X.shape[1]))

def build_model(train_generator):

    ''' buils model and prints out summary'''
    trainX, trainY = train_generator[0]

    learning_rate = 0.001
    units = 128
    epochs = 100
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU
    from tensorflow.keras.callbacks import EarlyStopping

    model = Sequential()
    model.add(tf.keras.layers.Masking(mask_value=10., input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(LSTM(units, ))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    adam = tf.keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    print(model.summary())

    print('Model Built')

    return model

def remove_padding(rows_to_remove, data, i):
    ''' removes padded values from the end of predicted arrays.
        can ignore '''

    rows = rows_to_remove[i]
    r = range(len(data)-rows, len(data))
    d = np.delete(data, r)
    return d

def make_predictions(model, testX, testY, test_generator, rows_to_remove, scalers):

    ''' Makes predictions on new values. Plots the error from true values
    stores erros in csv.
    can ignore
     '''
    # pred = model.predict(testX)
    # predictions = pred.reshape(pred.shape[0], pred.shape[1])
    # predictions = scalers['scaler_lap_time_remaining'].inverse_transform(predictions)
    # truth = scalers['scaler_lap_time_remaining'].inverse_transform(testY)
    # columns = [f'lap_{i}' for i in range(len(pred))]
    # y = truth-predictions
    # labels=[]
    # df = pd.DataFrame(columns=columns)
    # for i in range(len(y)):
    #     err = remove_padding(rows_to_remove, y[i], i)
    #     print(err.shape)
    #     lap = 'lap_'+str(i)
    #     df[lap] = pd.Series(err)
    #     plt.plot(err)

    # for i in range(10):
    #     x,y = test_generator[i]
    #     p = model.predict(x)
    #     preds.append(p)

    preds = model.predict(test_generator)
    pred = scalers['scaler_lap_time_remaining'].inverse_transform(preds)
    testY = scalers['scaler_lap_time_remaining'].inverse_transform(testY)
    y = testY[0][:-5]
    print(y.shape)
    p = pred.ravel()
    print(p.shape)
    err = p-y
    print(err.shape)
    plt.plot(y)
    plt.plot(p)
    plt.legend(['y','p'])
    plt.show()

def train_model(model, train_generator):
    ''' training the model'''
    EPOCHS = 1
    callback = [EarlyStopping(monitor="loss", min_delta = 0.0001, patience = 10, mode = 'auto', restore_best_weights=True),
    ModelCheckpoint('generator_lstm.h5')]
    history = model.fit(train_generator, callbacks=callback, shuffle=False, epochs=EPOCHS, batch_size=1)
    return history, model

def stack_generators(train_gens, trainX, trainY):

    ''' Iterates over generators created in multi_generator function.
        Tries to store each mini sequence in a list for use in training.
        I think this is what I'm doing wrong as currently getting an error about shape.
        This is because the look back from the generator is 5 timesteps for the X values,
        whereas the Y values is just 1 timestep. Not sure how I fix this.
     '''

    data = []
    target = []
    for gen in train_gens:
        for i in range(len(gen)):
            x,y = gen[i]
            data.append(x[0])
            target.append(y)

    return data, target

def concat_data(trainX, testX, trainY, testY):
    trainX, testX, trainY, testY = np.concatenate(trainX), np.concatenate(testX), np.concatenate(trainY), np.concatenate(testY)
    return trainX, testX, trainY, testY

def remove_overlap(train_generator, test_generator):
    ''' finds if there is overlap between laps in the generator'''
    for i in range(len(train_generator)):
        x,y = train_generator[i]
        lap = x[0][:,-1]
        start = lap[0]
        changes = [not i==start for i in lap]
        if True in changes:
            print('Value changes!')
            print(lap)
            x[:,:,:] = 0.


    return train_generator

class sequence_generator(TimeseriesGenerator):

    def __init__(self, data, targets, length,
                 sampling_rate=1,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=128):

        if len(data) != len(targets):
            raise ValueError('Data and targets have to be' +
                             ' of same length. '
                             'Data length is {}'.format(len(data)) +
                             ' while target length is {}'.format(len(targets)))

        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size

        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size *
                                    self.stride, self.end_index + 1), self.stride)

        samples = np.array([self.data[row - self.length:row:self.sampling_rate]
                            for row in rows])
        targets = np.array([self.targets[row] for row in rows])

        if self.reverse:
            return samples[:, ::-1, ...], targets

        samples, targets = self.check_overlap(samples, targets)

        l = samples.shape[2]
        samples = samples[:,:,:l-1]

        return samples, targets

    def check_overlap(self, samples, targets):
        x = samples[0]
        lap_col = x[:,-1]
        start = lap_col[0]
        changes = [not i==start for i in lap_col]
        if True in changes:
            samples[:,:,:] = 10.
        return samples, targets


if __name__ == '__main__':

    # data = import_training_data()
    data = pd.read_csv('sample_data.csv')

    trainX, testX, trainY, testY, scalers, num_rows_to_add = format_data(data)

    train_X, test_X, train_Y, test_Y = concat_data(trainX, testX, trainY, testY)

    look_back = 5
    batch_size=1

    train_generator = sequence_generator(train_X, train_Y, length=look_back, sampling_rate=1, stride=1, batch_size=batch_size)

    test_generator = sequence_generator(test_X, test_Y, length=look_back, sampling_rate=1, stride=1, batch_size=batch_size)


    # train_generator, test_generator = single_generator(train_X, test_X, train_Y, test_Y)

    # train_gens, test_gens, = multi_generator(trainX, testX, trainY, testY)
    #
    # X_train, y_train = stack_generators(train_gens, trainX, trainY)
    #
    # X_test, y_test = stack_generators(test_gens, trainX, trainY)
    #
    print('generators built')


    print_setup(train_generator, test_generator)

    model = build_model(train_generator)

    history, model = train_model(model, train_generator)
    # model = tf.keras.models.load_model('generator_lstm.h5')

    make_predictions(model, testX, testY, test_generator, num_rows_to_add, scalers)
