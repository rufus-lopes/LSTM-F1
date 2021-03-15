import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import sqlite3
import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler


def training_data():

    dir = '../SQL_Data/constant_setup'
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

physical_devices = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


data = training_data()

# sessionUID = data.pop('sessionUID').to_numpy()
# columns = data.columns
# data = sc.fit_transform(data)
# new_df = pd.DataFrame(data, columns=columns)
# new_df['sessionUID'] = sessionUID
# data = new_df
# print(data.head())
# unpack and arrange data by laps with associated final_lap_time

session_groups = data.groupby('sessionUID')
array_data = []
target_data = []
print(data.info())
for s in list(session_groups.groups):
    session = session_groups.get_group(s)
    lap_groups = session.groupby('currentLapNum')
    for l in list(lap_groups.groups):
        lap = lap_groups.get_group(l)
        laps = lap.drop('sessionUID', axis=1)
        final_lap_time = laps.pop('lap_time_remaining')
        laps = laps.to_numpy()
        laps = laps[::5] # sequence sampling
        final_lap_time = final_lap_time.iloc[0]
        target_data.append(final_lap_time)
        array_data.append(laps)

print(array_data[0].shape)
max_timesteps = max(array_data, key=len).shape[0]

num_rows_to_add = [max_timesteps-l.shape[0] for l in array_data]

data_dict = {k:v for (k,v) in zip(target_data, array_data)} # dictionary with lap time as key and lap data as value


# array padding - currently only using minimum value required so all the same size but might be worth increasing
# to allow for very slow laps. Maybe to 10000 timesteps?

lap_data = []

for i in range(len(array_data)):
    arr = array_data[i]
    rows_to_add = num_rows_to_add[i]
    array_to_append = np.zeros((rows_to_add, array_data[0].shape[1]), dtype=float)
    new_array = np.vstack((arr, array_to_append))
    lap_data.append(new_array)

# train_test_split

X_train = lap_data[:70]
X_test = lap_data[70:]
y_train = target_data[:70]
y_test = target_data[70:]
X_train = np.array(X_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)




input_shape = lap_data[0].shape

print('input shape is {}'.format(input_shape))
# define model

model = keras.Sequential()

model.add(layers.LSTM(50, input_shape=input_shape, return_sequences=True))
model.add(layers.Dropout(0.3))
model.add(layers.LSTM(300, return_sequences = True,))
model.add(layers.Dropout(0.3))
model.add(layers.LSTM(300,))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(100, activation='tanh',))
model.add(layers.Dense(50, activation='tanh'))
model.add(layers.Dense(1))

print(model.summary())
optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.001,
                                        rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')

model.compile(
loss = 'mean_squared_error',
optimizer=optimiser,
)

checkpoint_cb = keras.callbacks.ModelCheckpoint('LSTM_model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
shape = X_train.shape
print(f'X Shape is {shape}')
# history = model.fit(x=X_train, y = y_train, epochs=100, validation_split=0.2, callbacks=[checkpoint_cb, early_stopping_cb], batch_size=1)

model = keras.models.load_model('LSTM_model.h5')

test_loss= model.evaluate(x=X_test, y=y_test)

print("Test loss: ", test_loss)

pred = model.predict(X_test, batch_size=1)
print(f'y train mean is {np.mean(y_train)}')
print(f'Prediction shape is {pred.shape}')
print(pred)
print(f'Y test shape is {y_test.shape}')
print(y_test)

# summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
