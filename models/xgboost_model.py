import pandas as pd
from import_data import import_training_data
import numpy as np
from sklearn.model_selection import train_test_split
import time
import xgboost
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,KFold

df = import_training_data()
label = df.pop('finalLapTime')

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.4, random_state=42)


model = xgboost.XGBRegressor()

model.fit(X_train, y_train)
pred = model.predict(X_test)

print(mae(y_test, pred))
