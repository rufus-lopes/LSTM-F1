
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import pickle
from import_data import import_training_data
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels as sm

def save_model(model):
    file_name = 'LinearRegression.pkl'
    pickle.dump(model, open(file_name, 'wb'))

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

#format training data
df = import_training_data()
labels = df.pop('finalLapTime')
names = df.columns
X, x, Y, y = train_test_split(df, labels, test_size=0.2, random_state=42)

# initialise and train model
model = LinearRegression()
model.fit(X, Y)
preds = model.predict(x)
coefs = model.coef_
save_model(model)
beta_coefficients = {key:value for (key,value) in zip(names,coefs)}
sorted_beta_coefficients = dict(sorted(beta_coefficients.items(), key=lambda item: item[1]))
correlations = df.corr()
sns.heatmap(correlations, xticklabels=correlations.columns, yticklabels=correlations.columns, cmap='RdBu')
plt.show()


x1 = add_constant(df)
vif = pd.DataFrame()
vif['variable'] = df.columns
vif['VIF'] = [variance_inflation_factor(x1.values, i) for i in range(x1.shape[1])]

print(vif)


# evaluate predictions
# print(mae(y, preds))
# print(preds[:5])
