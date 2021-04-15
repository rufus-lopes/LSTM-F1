import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def get_data():
    df = pd.read_csv('../../prediction_csv/XGBoost_walk_predictions.csv')
    df.drop([df.columns[0]], inplace=True, axis=1)
    return df

def get_slowest_lap(df):
    slowest = []
    sessions = df.groupby('sessionUID')
    for s in list(sessions.groups):
        session = sessions.get_group(s)
        laps = session.groupby('truth')
        slowest.append(laps.get_group(max(list(laps.groups))))


    return slowest


def plot_prediction(df):
    fig , (ax1, ax2) = plt.subplots(1, 2)
    dy = 6*np.std(df['residuals'])
    df = df.drop(df[abs(df.residuals)>dy].index)
    ax1.plot(df['truth'])
    ax1.plot(df['predictions'])

    ax2.scatter(x=df['worldPositionX'], y = df['worldPositionZ'],c=df['currentLapTime'], s=1)
    plt.show()

if __name__ == '__main__':
    df = get_data()

    lap = get_slowest_lap(df)
    i = 0
    for l in lap:
        plot_prediction(l)
        i+=1
        if i == 5:
            break
