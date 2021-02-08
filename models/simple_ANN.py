import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from import_data import import_training_data
from sklearn.model_selection import train_test_split
import os
import time

df = import_training_data()
target = df.pop('finalLapTime')
X_train, X_test, y_train, y_test = train_test_split(df, target)

dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
train_dataset = dataset.shuffle(len(X_train)).batch(1)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.shuffle(len(X_test)).batch(1)

root_logdir = os.path.join(os.curdir, 'my_logs')

def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d_%H_%M_%S')
    return os.path.join(root_logdir, run_id)


NAME = 'simple_ANN_{}'.format(int(time.time()))
run_logdir = get_run_logdir()


# X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

model = keras.models.Sequential()

model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')


tensorboard_cb = keras.callbacks.TensorBoard(log_dir='test_logs/{}'.format(NAME))
checkpoint_cb = keras.callbacks.ModelCheckpoint('simple_ANN.h5')

history = model.fit(train_dataset, epochs=20,validation_data=test_dataset, callbacks=[checkpoint_cb])
