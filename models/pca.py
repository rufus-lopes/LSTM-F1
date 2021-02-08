import pandas as pd
from import_data import import_training_data
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

df = import_training_data()
y = df.pop('finalLapTime')
X_train, X_test, y_train, y_test = train_test_split(df,y)
pca = PCA()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
pca.fit_transform(df)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum>0.95)+1
pca = PCA(n_components=d)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(mae(y_test, pred))
