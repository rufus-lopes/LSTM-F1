import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import os
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from statsmodels.stats.diagnostic import normal_ad

plt.style.use('seaborn-whitegrid')


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

    sector_1_r2 = round(r_squared(sector_1),3)
    sector_2_r2 = round(r_squared(sector_2),3)
    sector_3_r2 = round(r_squared(sector_3),3)

    return sector_1_r2, sector_2_r2, sector_3_r2


def plot_residuals(df):
    x=df['currentLapTime']
    y=df['residuals']
    dy = 3*np.std(df['residuals'])

    for i in range(len(y)):
        if abs(y[i]) > dy:
            y.pop(i)
            x.pop(i)

    r_2 = round(r_squared(df), 3)
    plt.figure(figsize=(15, 8))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().xaxis.label.set_size('20')
    # plt.gca().yaxis.label.set_size('20')
    # plt.gca().title.label.set_size('20')
    plt.scatter(x=x, y=y, s=20, color='xkcd:lightish blue', alpha=0.8, zorder=1)
    # plt.fill_between(x, y - dy, y + dy, color='gray', alpha=0.2)
    # plt.title('Prediction Residuals for SVR Model', fontsize=30, fontweight='bold')
    plt.xlabel('Current Lap Time (s)', fontsize=30, fontweight='bold')
    plt.ylabel('Residuals', fontsize=30, fontweight='bold')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.text(100, 10 ,f'R2: {r_2}', fontsize=25, fontweight='bold')
    plt.axhline(y=0,  color ="green", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    plt.savefig('SVR_pca_residuals.png', bbox_inches='tight',pad_inches = 0)
    plt.show()


def group_by_sector(df):

    sectors = df.groupby('sector')

    for s in list(sectors.groups):
        sector = sectors.get_group(s)
        if s == 0:
            sector_1 = sector
        elif s == 1:
            sector_2 = sector
        elif s == 2:
            sector_3 = sector

    return sector_1, sector_2, sector_3

def plot_residuals_by_sector(df):

    sector_1_r2, sector_2_r2, sector_3_r2 = r_squared_by_sector(df)

    dy = 3*np.std(df['residuals'])
    df = df.drop(df[abs(df.residuals)>dy].index)

    sector_1, sector_2, sector_3 = group_by_sector(df)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize=(25,15))

    # fig.suptitle('Prediction Residuals Grouped by Sector for SVR Model', fontsize=30, fontweight='bold')

    sector1_x, sector1_y = sector_1['currentLapTime'], sector_1['residuals']
    sector2_x, sector2_y = sector_2['currentLapTime'], sector_2['residuals']

    sector3_x, sector3_y = sector_3['currentLapTime'], sector_3['residuals']

    ax1.scatter(x = sector1_x , y = sector1_y,s=20, color='orange', alpha=0.8, zorder=1)
    ax2.scatter(x = sector2_x, y = sector2_y,s=20, color='darkcyan', alpha=0.8, zorder=1)
    ax3.scatter(x = sector3_x, y = sector3_y,s=20, color='indianred', alpha=0.8, zorder=1)

    ax1.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    ax2.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    ax3.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)

    ax1.text(20, 10, f'R2: {sector_1_r2}', fontsize=40, fontweight='bold')
    ax2.text(50, 10, f'R2: {sector_2_r2}', fontsize=40, fontweight='bold')
    ax3.text(80, 10, f'R2: {sector_3_r2}', fontsize=40, fontweight='bold')

    ax1.tick_params(axis="x", labelsize=30)
    ax2.tick_params(axis="x", labelsize=30)
    ax3.tick_params(axis="x", labelsize=30)

    ax1.tick_params(axis="y", labelsize=30)
    ax2.tick_params(axis="y", labelsize=30)
    ax3.tick_params(axis="y", labelsize=30)

    ax1.set_title('Sector 1', fontsize=40, fontweight='bold')
    ax2.set_title('Sector 2', fontsize=40, fontweight='bold')
    ax3.set_title('Sector 3', fontsize=40, fontweight='bold')

    ax2.set_xlabel('Current Lap Time (s)', fontsize=40, fontweight='bold',)
    ax1.set_ylabel('Residuals', fontsize=40, fontweight='bold',)

    plt.savefig('SVR_pca_residuals_by_sector.png', bbox_inches='tight',pad_inches = 0)

    plt.show()



if __name__ == '__main__':
    svr = pd.read_csv('../../prediction_csv/SVR_pca_predictions.csv')
    svr.drop([svr.columns[0]], axis=1, inplace=True)

    plot_residuals(svr)

    svr = pd.read_csv('../../prediction_csv/SVR_pca_predictions.csv')
    svr.drop([svr.columns[0]], axis=1, inplace=True)


    plot_residuals_by_sector(svr)
