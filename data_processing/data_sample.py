from import_data import import_training_data


data = import_training_data()

df = data.head(20000)

df.to_csv('Bayesian_RNN/sample_data.csv')

print(df.info())
