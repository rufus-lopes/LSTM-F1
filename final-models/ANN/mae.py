import pandas as pd
from sklearn.metrics import mean_absolute_error as mae

df = pd.read_csv('ANN_predictions.csv')

truth = df['finalLapTime']
pred = df['predictions']

print(mae(truth, pred))
