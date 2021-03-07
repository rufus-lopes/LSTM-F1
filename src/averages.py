import sqlite3
import pandas as pd
import numpy as np

def averages(df, timeStep):
    '''takes in a dataframe and timeStep and returns the a dataframe
    of the rolling average over that timeStep '''
    roll = df.rolling(timeStep, min_periods=1).mean()#maybe needs some tweaking
    return roll

def sums(df):
    '''takes the averaged df as input and concatonates onto the side
    the cumulative sum of all variables'''

    sumDf = df.cumsum()
    return sumDf

def getMainDf(db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute('SELECT * FROM MasterData')
    df = pd.DataFrame(cur.fetchall())
    names = list(map(lambda x: x[0], cur.description))
    df.columns = names
    df.set_index(names[0], inplace=True)
    names.pop(0)
    return df, names, conn

def getIsPit(db):
    '''checks to see whether the car has pitted in laps'''
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute('SELECT currentLapNum, pitStatus from LapData')
    df = pd.DataFrame(cur.fetchall())
    names = list(map(lambda  x: x[0], cur.description))
    df.columns = names
    g = df.groupby('currentLapNum')
    lap = []
    isPit = []
    for i in list(g.groups):
        l = g.get_group(i)
        pitStatus = l['pitStatus'].unique()
        if len(pitStatus) == 1:
            isPit.append(False)
        else:
            isPit.append(True)
        lap.append(i)
    pitLap = {l:p for (l,p) in zip(lap, isPit)}
    return pitLap

def groupByLaps(df):
    """groups data by laps and returns grouped data in a list """
    g = df.groupby('currentLapNum')
    groupNames = list(g.groups)
    data = []
    for n in groupNames:
        data.append(g.get_group(n))
    return data
1
def selectSumColumns(df, columnsToSum):
    '''selects appropriate columns to be summed'''
    df = df[columnsToSum]
    return df

def toSQL(df, conn):
    df.to_sql('TrainingData', con = conn, schema = None, if_exists = 'replace')

def checkFullLap(df, isPit):
    g = df.groupby('currentLapNum')
    groupNames = list(g.groups)
    data = []
    for n in groupNames:
        lap = g.get_group(n)
        finalTime = lap['finalLapTime'].to_numpy()
        if finalTime[0] > 80 and not isPit[n]:
            data.append(lap)
    if data:
        return pd.concat(data)
def get_lap_time_remaining(df):
    g = df.groupby('currentLapNum')
    data = []
    groupNames = list(g.groups)
    for n in groupNames:
        lap = g.get_group(n)
        lap['lap_time_remaining'] = lap['finalLapTime'] - lap['currentLapTime']
        data.append(lap)
    if data:
        return pd.concat(data)

def trainingCalculations(db):

    columnsToSum = [
    'currentLapTime', 'worldPositionX', 'worldPositionY', 'worldPositionZ',
    'worldVelocityX', 'worldVelocityY', 'worldVelocityZ', 'yaw', 'pitch',
    'roll', 'speed', 'throttle', 'steer', 'brake', 'gear',
    'engineRPM', 'drs', 'brakesTemperatureRL', 'brakesTemperatureRR',
    'brakesTemperatureFL', 'brakesTemperatureFR',  'tyresSurfaceTemperatureRL',
    'tyresSurfaceTemperatureRR', 'tyresSurfaceTemperatureFL', 'tyresSurfaceTemperatureFR',
    'engineTemperature','tyresWearRL', 'tyresWearRR', 'tyresWearFL', 'tyresWearFR', 'carPosition'
    ]

    df, names, conn = getMainDf(db)
    # timeStep = 10 # currently measured in packets - can easily adjust from here
    # data = groupByLaps(df)
    # sessionUID = df['sessionUID']
    #
    # averagedData = [averages(d, timeStep) for d in data]# perform averaging on each lap individually
    # fullAveragedData = pd.concat(averagedData) # merge averaged laps into a single df
    #
    # summedData = [sums(selectSumColumns(i, columnsToSum)) for i in data]#calculate cumulative sum on each lap
    # sumNames = ['summed_'+ name for name in columnsToSum] #modify names for summed variables
    # fullSummedData = pd.concat(summedData)#merge to single df
    # fullSummedData.columns = sumNames #attach correct names
    #
    # finalData = pd.concat([fullAveragedData, fullSummedData], axis = 1, ignore_index=False) #merge to create unified training data df
    finalData=df
    is_pit = getIsPit(db)
    finalData = checkFullLap(finalData, is_pit)
    finalData = get_lap_time_remaining(finalData)
    # finalData['sessionUID'] = sessionUID
    toSQL(finalData, conn) # send data back to SQL file
