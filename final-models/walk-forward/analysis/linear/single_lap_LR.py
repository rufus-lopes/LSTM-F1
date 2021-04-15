import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def get_data():
    df = pd.read_csv('../../prediction_csv/Linear_walk_predictions.csv')
    df.drop([df.columns[0]], inplace=True, axis=1)
    return df

def get_slowest_lap(df):
    slowest = {}
    sessions = df.groupby('sessionUID')
    for s in list(sessions.groups):
        session = sessions.get_group(s)
        laps = session.groupby('truth')
        slowest[max(list(laps.groups))] = laps.get_group(max(list(laps.groups)))

    return slowest[max(slowest)]




def plot_prediction(df):
    fig , (ax1, ax2) = plt.subplots(1, 2)
    dy = 3*np.std(df['residuals'])
    df = df.drop(df[abs(df.residuals)>dy].index)
    ax1.plot(df['truth'])
    ax1.plot(df['predictions'])

    ax2.scatter(x=df['worldPositionX'], y = df['worldPositionZ'],c=df['speed'], s=1)
    plt.show()

if __name__ == '__main__':
    df = get_data()

    lap = get_slowest_lap(df)

    plot_prediction(lap)
