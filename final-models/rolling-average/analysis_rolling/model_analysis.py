import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import os
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

def r_squared(df):
    truth = df['truth']
    pred = df['predictions']
    return r2_score(truth, pred)


def r_squared_by_sector(df):
    sector_1 = []
    sector_2 = []
    sector_3 = []
    sessions = df.groupby('sessionUID')
    for s in list(sessions.groups):
        session = sessions.get_group(s)
        laps = session.groupby('currentLapNum')
        for l in list(laps.groups):
            lap = laps.get_group(l)
            sectors = lap.groupby('sector')
            for j in list(sectors.groups):
                sector = sectors.get_group(j)
                if j == 0:
                    sector_1.append(sector)
                elif j == 1:
                    sector_2.append(sector)
                elif j == 2:
                    sector_3.append(sector)

    sector_1 = pd.concat(sector_1)
    sector_2 = pd.concat(sector_2)
    sector_3 = pd.concat(sector_3)

    sector_1_r2 = r_squared(sector_1)
    sector_2_r2 = r_squared(sector_2)
    sector_3_r2 = r_squared(sector_3)

    return sector_1_r2, sector_2_r2, sector_3_r2



if __name__ == '__main__':

    # svr_sample = pd.read_csv('prediction_csv/SVR_sample_predictions.csv')
    # r2_svr = r_squared(svr_sample)
    # print(f'SVR R squared: {r2_svr}')
    #



    # xgb = pd.read_csv('prediction_csv/XGBoost_predictions.csv')


    #
    #
    #
    # sns.set_theme(color_codes=True)
    # g = sns.lmplot(x="total_bill", y="tip", data=tips)
