import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import os
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from statsmodels.stats.diagnostic import normal_ad

plt.style.use('seaborn')


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
    plt.scatter(x=x, y=y, s=0.2, color='xkcd:lightish blue', alpha=0.5, zorder=1)
    # plt.fill_between(x, y - dy, y + dy, color='gray', alpha=0.2)
    # plt.title('Prediction Residuals for Linear Model', fontsize=30, fontweight='bold')
    plt.xlabel('Current Lap Time (s)', fontsize=30, fontweight='bold')
    plt.ylabel('Residuals', fontsize=30, fontweight='bold')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.text(100, 10 ,f'R2: {r_2}', fontsize=25, fontweight='bold')
    plt.axhline(y=0,  color ="green", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    plt.savefig('linear_residuals.png')
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

def plot_residuals_by_sector(linear, svr, xgb):

    s1_lin_r2, s2_lin_r2, s3_lin_r2 = r_squared_by_sector(linear)
    s1_svr_r2, s2_svr_r2, s3_svr_r2 = r_squared_by_sector(svr)
    s1_xgb_r2, s2_xgb_r2, s3_xgb_r2 = r_squared_by_sector(xgb)

    dy_lin = 3*np.std(linear['residuals'])
    dy_svr = 3*np.std(svr['residuals'])
    dy_xgb = 3*np.std(xgb['residuals'])

    linear = linear.drop(linear[abs(linear.residuals)>dy_lin].index)
    svr = svr.drop(svr[abs(svr.residuals)>dy_svr].index)
    xgb = xgb.drop(xgb[abs(xgb.residuals)>dy_xgb].index)



    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3, figsize=(25,15), sharex='col', sharey='row')

    lin_sector_1, lin_sector_2, lin_sector_3 = group_by_sector(linear)
    svr_sector_1, svr_sector_2, svr_sector_3 = group_by_sector(svr)
    xgb_sector_1, xgb_sector_2, xgb_sector_3 = group_by_sector(xgb)

    ax1.scatter(x = lin_sector_1['currentLapTime'] , y = lin_sector_1['residuals'],s=5, color='orange', alpha=0.5, zorder=1)
    ax2.scatter(x = lin_sector_2['currentLapTime'], y = lin_sector_2['residuals'],s=5, color='darkcyan', alpha=0.5, zorder=1)
    ax3.scatter(x = lin_sector_3['currentLapTime'], y = lin_sector_3['residuals'],s=5, color='indianred', alpha=0.5, zorder=1)

    ax4.scatter(x = svr_sector_1['currentLapTime'], y = svr_sector_1['residuals'],s=5, color='orange', alpha=0.5, zorder=1)
    ax5.scatter(x = svr_sector_2['currentLapTime'], y = svr_sector_2['residuals'],s=5, color='darkcyan', alpha=0.5, zorder=1)
    ax6.scatter(x = svr_sector_3['currentLapTime'], y = svr_sector_3['residuals'],s=5, color='indianred', alpha=0.5, zorder=1)

    ax7.scatter(x = xgb_sector_1['currentLapTime'], y = xgb_sector_1['residuals'],s=5, color='orange', alpha=0.5, zorder=1)
    ax8.scatter(x = xgb_sector_2['currentLapTime'], y = xgb_sector_2['residuals'],s=5, color='darkcyan', alpha=0.5, zorder=1)
    ax9.scatter(x = xgb_sector_3['currentLapTime'], y = xgb_sector_3['residuals'],s=5, color='indianred', alpha=0.5, zorder=1)

    ax1.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    ax2.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    ax3.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    ax4.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    ax5.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    ax6.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    ax7.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    ax8.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)
    ax9.axhline(y=0,  color ="midnightblue", linestyle ="--", linewidth=3,label='Target Value', zorder=2)

    # ax1.axhline(y=max(linear['residuals']),  color ="midnightblue", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    # ax2.axhline(y=max(linear['residuals']),  color ="midnightblue", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    # ax3.axhline(y=max(linear['residuals']),  color ="midnightblue", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    ax4.axhline(y=max(svr['residuals']),  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    ax5.axhline(y=max(svr['residuals']),  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    ax6.axhline(y=max(svr['residuals']),  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    ax7.axhline(y=max(xgb['residuals']),  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    ax8.axhline(y=max(xgb['residuals']),  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    ax9.axhline(y=max(xgb['residuals']),  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)

    ax1.axvline(x=70,  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    ax4.axvline(x=70,  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    ax7.axvline(x=70,  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    ax2.axvline(x=110,  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    ax5.axvline(x=110,  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)
    ax8.axvline(x=110,  color ="black", linestyle ="-", linewidth=1,label='Target Value', zorder=2)

    ax1.text(35, 10, f'R2: {s1_lin_r2}', fontsize=30, fontweight='bold')
    ax2.text(75, 10, f'R2: {s2_lin_r2}', fontsize=30, fontweight='bold')
    ax3.text(105, 10, f'R2: {s3_lin_r2}', fontsize=30, fontweight='bold')
    ax4.text(35, 10, f'R2: {s1_svr_r2}', fontsize=30, fontweight='bold')
    ax5.text(75, 10, f'R2: {s2_svr_r2}', fontsize=30, fontweight='bold')
    ax6.text(105, 10, f'R2: {s3_svr_r2}', fontsize=30, fontweight='bold')
    ax7.text(35, 0.1, f'R2: {s1_xgb_r2}', fontsize=30, fontweight='bold')
    ax8.text(75, 0.1, f'R2: {s2_xgb_r2}', fontsize=30, fontweight='bold')
    ax9.text(105, 0.1, f'R2: {s3_xgb_r2}', fontsize=30, fontweight='bold')

    ax1.set_ylabel('LR',  fontsize=40, fontweight='bold')
    ax4.set_ylabel('SVR',  fontsize=40, fontweight='bold')
    ax7.set_ylabel('XGB',  fontsize=40, fontweight='bold')

    ax1.tick_params(axis="y", labelsize=30)
    ax4.tick_params(axis="y", labelsize=30)
    ax7.tick_params(axis="y", labelsize=30)

    ax7.tick_params(axis="x", labelsize=30)
    ax8.tick_params(axis="x", labelsize=30)
    ax9.tick_params(axis="x", labelsize=30)

    ax7.set_xlabel('Sector 1', fontsize=40, fontweight='bold')
    ax8.set_xlabel('Sector 2', fontsize=40, fontweight='bold')
    ax9.set_xlabel('Sector 3', fontsize=40, fontweight='bold')

    plt.subplots_adjust(wspace=.0, hspace=.0,)
    plt.savefig('full_comparison.png', bbox_inches='tight',pad_inches = 0)
    plt.show()


if __name__ == '__main__':

    linear = pd.read_csv('../prediction_csv/Linear_walk_predictions.csv')
    linear.drop([linear.columns[0]], axis=1, inplace=True)

    svr = pd.read_csv('../prediction_csv/SVR_pca_predictions.csv')
    svr.drop([svr.columns[0]], axis=1, inplace=True)

    xgb = pd.read_csv('../prediction_csv/XGBoost_walk_predictions.csv')
    xgb.drop([xgb.columns[0]], axis=1, inplace=True)

    plot_residuals_by_sector(linear, svr, xgb)

    # plot_residuals(linear)

    # plot_residuals_by_sector(linear)

    # normal_errors_assumption(linear)
