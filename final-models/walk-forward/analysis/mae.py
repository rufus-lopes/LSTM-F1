import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae


def get_mae(df):
    truth = df['truth']
    pred = df['predictions']
    return mae(truth, pred)


def mae_by_sector(df):
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

    truth_1 = sector_1['truth']
    pred_1  = sector_1['predictions']
    truth_2 = sector_2['truth']
    pred_2  = sector_2['predictions']
    truth_3 = sector_3['truth']
    pred_3  = sector_3['predictions']

    sector_1_mae = round(mae(truth_1, pred_1),3)
    sector_2_mae = round(mae(truth_2, pred_2),3)
    sector_3_mae = round(mae(truth_3, pred_3),3)

    return sector_1_mae, sector_2_mae, sector_3_mae



if __name__ == '__main__':
    lin = pd.read_csv('../prediction_csv/Linear_walk_predictions.csv')
    lin = lin.drop([lin.columns[0]],axis=1)

    svr_sub = pd.read_csv('../prediction_csv/SVR_sample_predictions.csv')
    svr_sub = svr_sub.drop([svr_sub.columns[0]],axis=1)

    svr_pca = pd.read_csv('../prediction_csv/SVR_pca_predictions.csv')
    svr_pca = svr_pca.drop([svr_pca.columns[0]],axis=1)

    xgb = pd.read_csv('../prediction_csv/XGBoost_walk_predictions.csv')
    xgb = xgb.drop([xgb.columns[0]],axis=1)


    lin_mae = get_mae(lin)
    svr_sub_mae = get_mae(svr_sub)
    svr_pca_mae = get_mae(svr_pca)
    xgb_mae = get_mae(xgb)

    print(f'Linear MAE: {lin_mae}')
    print(f'Sub sampled SVR MAE: {svr_sub_mae}')
    print(f'PCA SVR MAE: {svr_pca_mae}')
    print(f'XGB MAE: {xgb_mae}')


    sector_1_mae_lin, sector_2_mae_lin, sector_3_mae_lin = mae_by_sector(lin)
    sector_1_mae_svr_sub, sector_2_mae_svr_sub, sector_3_mae_svr_sub = mae_by_sector(svr_sub)
    sector_1_mae_svr_pca, sector_2_mae_svr_pca, sector_3_mae_svr_pca = mae_by_sector(svr_pca)
    sector_1_mae_xgb, sector_2_mae_xgb, sector_3_mae_xgb = mae_by_sector(xgb)

    print('linear: ')
    print(sector_1_mae_lin, sector_2_mae_lin, sector_3_mae_lin)
    print('*********************')
    print('svr sub')
    print(sector_1_mae_svr_sub, sector_2_mae_svr_sub, sector_3_mae_svr_sub)
    print('*********************')
    print('svr pca')
    print(sector_1_mae_svr_pca, sector_2_mae_svr_pca, sector_3_mae_svr_pca)
    print('*********************')
    print('XGBoost')
    print(sector_1_mae_xgb, sector_2_mae_xgb, sector_3_mae_xgb)
    print("*********************")
