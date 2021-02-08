import tensorflow as tf
import pandas as pd
from tensorflow import keras
from import_data import import_training_data
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
root_logdir = os.path.join(os.curdir, 'my_logs')

def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d_%H_%M_%S')
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

df = import_training_data()
y = df.pop('finalLapTime')

df = df.to_numpy()
y = y.to_numpy()
s = df.shape
df= df.reshape(s[0], s[1], 1)

print(df[0])

X_train = df
y_train = y


X_train, X_test, y_train, y_test = train_test_split(df, y)

model = keras.Sequential()

model.add(layers.SimpleRNN(128, return_sequences = True, input_shape=[None,1]))
model.add(layers.SimpleRNN(128, return_sequences = True))
model.add(layers.Dense(1))

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
model.fit(x=X_train,y=y_train,callbacks=[tensorboard_cb], epochs=5)
preds = model.predict(X_test)
print(preds)
print(mean_absolute_error(y_test, preds))
