import tensorflow as tf
from tensorflow import keras
from import_data import import_training_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
model = keras.models.load_model('simple_ANN.h5')

data = import_training_data()
target = data.pop('finalLapTime')

X_train, X_test, y_train, y_test = train_test_split(data, target)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.shuffle(len(X_test)).batch(1)

pred = model.predict(test_dataset)

print(f'Mean average error on 33% test set: {mae(y_test, pred)}')
