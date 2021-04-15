import pandas as pd
import sqlite3
import os
import dask.dataframe as dd


def sub_sample(df):
    df2 = df[df.index % 10 == 0]
    return df2


def get_all_data():

    dir = '../../SQL_Data/constant_setup'
    files = os.listdir(dir)
    files = [f for f in files if f.endswith('.sqlite3')]

    motion = []
    lap = []
    telemetry = []
    status = []

    for f in files:
        path = os.path.join(dir, f)
        conn = sqlite3.connect(path)
        if os.path.getsize(path) > 10000:
            con = sqlite3.connect(path)
            cursor = con.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table in tables:
                table = table[0]
                if table == 'motionData':
                    cursor.execute(f'SELECT * FROM {table}')
                    motion.append(pd.DataFrame(cursor.fetchall()))
                    motion_names = list(map(lambda x: x[0], cursor.description))
                if table == 'LapData':
                    cursor.execute(f'SELECT * FROM {table}')
                    lap.append(pd.DataFrame(cursor.fetchall()))
                    lap_names = list(map(lambda x: x[0], cursor.description))
                if table == 'TelemetryData':
                    cursor.execute(f'SELECT * FROM {table}')
                    telemetry.append(pd.DataFrame(cursor.fetchall()))
                    telemetry_names = list(map(lambda x: x[0], cursor.description))
                if table == 'CarStatusData':
                    cursor.execute(f'SELECT * FROM {table}')
                    status.append(pd.DataFrame(cursor.fetchall()))
                    status_names = list(map(lambda x: x[0], cursor.description))


    print(f'{len(motion)} SQL files captured')

    motion = pd.concat(motion)
    lap = pd.concat(lap)
    telemetry = pd.concat(telemetry)
    status = pd.concat(status)

    motion.columns = motion_names
    lap.columns = lap_names
    telemetry.columns = telemetry_names
    status.columns = status_names

    motion = sub_sample(motion)
    lap = sub_sample(lap)
    telemetry = sub_sample(telemetry)
    status = sub_sample(telemetry)

    return motion, lap, telemetry, status

def merge_dfs(motion, lap, telemetry, status):
    master = motion.merge(lap, suffixes=('', '_y'), on='frameIdentifier')
    master = master.merge(telemetry,suffixes=('', '_y'),  on='frameIdentifier')
    master = master.merge(status,suffixes=('', '_y'),  on = 'frameIdentifier')
    return master

def clean_master(df):
    cols_to_drop = [x for x in list(df.columns) if x.endswith(('_y',)) ]
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

def finalLapTime(df):
    lapNums = df['currentLapNum']
    lapTimes = df['currentLapTime']
    newLapIndex = []
    for i in range(len(lapNums)-1):
        currentLap = lapNums[i]
        nextLap = lapNums[i+1]
        if currentLap != nextLap:
            newLapIndex.append(i)
    newLapIndex.append(i+1)
    finalLapTimes = {lap:time for (lap,time) in zip(lapNums[newLapIndex],lapTimes[newLapIndex])}
    df["finalLapTime"] = [finalLapTimes[lap] for lap in lapNums]
    return df

def get_lap_time_remaining(df):
    data = []
    sessions = df.groupby('sessionUID')
    for s in list(sessions.groups):
        sess = sessions.get_group(s)
        g = sess.groupby('currentLapNum')
        groupNames = list(g.groups)
        for n in groupNames:
            lap = g.get_group(n)
            lap['lap_time_remaining'] = lap['finalLapTime'] - lap['currentLapTime']
            data.append(lap)
    if data:
        return pd.concat(data)


if __name__ == '__main__':

    motion, lap, telemetry, status = get_all_data()

    master  = merge_dfs(motion, lap, telemetry, status)

    master = clean_master(master)

    master = finalLapTime(master)

    master = get_lap_time_remaining(master)

    master.to_csv('full_data.csv')
