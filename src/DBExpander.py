import sqlite3
import pandas as pd
import struct
from src.datatypes import PacketID, EventStringCode
from src.UDP_unpacker import unpackUDPpacket
import os
import inspect
import ctypes
from src.databaseUnpacker import localFormat
import warnings
import numpy as np
import matplotlib
from src.addColumnNames import addColumnNames
import logging
from src.times import *
from src.averages import trainingCalculations

warnings.filterwarnings("ignore")

def connect(db_name):
    conn = sqlite3.connect(db_name)
    return conn

def selectALL(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM packets ORDER BY pkt_id ASC")
    #cur.execute("PRAGMA table_info(packets)") #just checking table column names
    rows = cur.fetchall()
    df = pd.DataFrame(rows)
    return df


def globalFormater(df):
    packetTypes = df["packetId"]
    packets = df["packet"]
    packetIDs = df["pkt_id"]
    sessionIDs = df["sessionUID"]
    motionArr = []
    sessionArr = []
    lapDataArr = []
    eventArr = []
    participantsArr = []
    carSetupsArr = []
    carTelemetryArr = []
    carStatusArr =[]
    finalClassificationArr = []
    lobbyInfoArr = []

    for i in range(len(packetTypes)):
        packetType = packetTypes[i]
        packetIDrelation = packetIDs[i]
        packetSessionRelation = sessionIDs[i]
        packet = unpackUDPpacket(packets[i])
        formatted_info = localFormat(packet, packetType)

        if packetType == 0:
            motionArr.append(formatted_info.arr)
        elif packetType == 1:
            sessionArr.append(formatted_info.arr)
        elif packetType == 2:
            lapDataArr.append(formatted_info.arr)
        elif packetType == 3:
            eventArr.append(formatted_info.arr)
        elif packetType == 4:
            participantsArr.append(formatted_info.arr)
        elif packetType == 5:
            carSetupsArr.append(formatted_info.arr)
        elif packetType == 6:
            carTelemetryArr.append(formatted_info.arr)
        elif packetType == 7:
            carStatusArr.append(formatted_info.arr)
        elif packetType == 8:
            finalClassificationArr.append(formatted_info.arr)
        elif packetType == 9:
            lobbyInfoArr.append(formatted_info.arr)

    motionDF = pd.DataFrame(motionArr)
    sessionDF = pd.DataFrame(sessionArr)
    lapDataDF = pd.DataFrame(lapDataArr)
    eventDF = pd.DataFrame(eventArr)
    participantsDF = pd.DataFrame(participantsArr)
    carSetupsDF = pd.DataFrame(carSetupsArr)
    carTelemetryDF = pd.DataFrame(carTelemetryArr)
    carStatusDF = pd.DataFrame(carStatusArr)
    finalClassificationDF = pd.DataFrame(finalClassificationArr)
    lobbyInfoDF = pd.DataFrame(lobbyInfoArr)

    motionDF, sessionDF, lapDataDF, eventDF, carSetupsDF, carTelemetryDF, carStatusDF = addColumnNames(motionDF,
    sessionDF, lapDataDF, eventDF, carSetupsDF, carTelemetryDF, carStatusDF)

    return motionDF, sessionDF, lapDataDF, eventDF, carSetupsDF, carTelemetryDF, carStatusDF

def Tables(motionDF, sessionDF, lapDataDF, eventDF, carSetupsDF, carTelemetryDF, carStatusDF, conn):
    motionDF.to_sql("motionData", con = conn, schema=None, if_exists="replace")
    lapDataDF.to_sql("LapData", con = conn, schema=None, if_exists="replace")
    eventDF.to_sql("EventData", con = conn, schema=None, if_exists="replace")
    carSetupsDF.to_sql("CarSetupData", con = conn, schema=None, if_exists="replace")
    carTelemetryDF.to_sql("TelemetryData", con = conn, schema=None, if_exists="replace")
    carStatusDF.to_sql("CarStatusData", con = conn, schema=None, if_exists="replace")
    sessionDF.to_sql("SessionData", con=conn, schema=None, if_exists="replace")

#getting master table
def masterLapData(connection):
    lapTimesDf = getLapTimes(connection)
    lapTimesDf = addNames(lapTimesDf)
    lapTimesDf = finalLapTime(lapTimesDf)
    masterLapDataDfVars = ["frameIdentifier", "lastLapTime", "currentLapTime", "bestLapTime", "currentLapNum", "finalLapTime", "lapDistance", "carPosition"]
    masterLapDataDf = lapTimesDf[masterLapDataDfVars]
    return masterLapDataDf


def masterPacketData(connection):
    cur = connection.cursor()
    cur.execute("SELECT frameIdentifier, pkt_id, packetId, sessionUID, sessionTime FROM packets")
    masterPacketDf = pd.DataFrame(cur.fetchall())
    packetCols = ["frameIdentifier", "pkt_id", "packetId", "sessionUID", "SessionTime"]
    masterPacketDf.columns = packetCols
    return masterPacketDf

def masterMotionData(connection):
    cur = connection.cursor()
    cur.execute("""SELECT frameIdentifier, worldPositionX, worldPositionY, worldPositionZ,
        worldVelocityX, worldVelocityY, worldVelocityZ, yaw, pitch, roll FROM motionData""")

    masterMotionDf = pd.DataFrame(cur.fetchall())
    motionCols = ["frameIdentifier", "worldPositionX", "worldPositionY", "worldPositionZ",
        "worldVelocityX", "worldVelocityY", "worldVelocityZ", "yaw", "pitch", "roll"]

    masterMotionDf.columns = motionCols
    return masterMotionDf

def masterTelemetryData(connection):
    cur = connection.cursor()
    cur.execute("""SELECT frameIdentifier, speed, throttle, steer, brake, clutch, gear, engineRPM,
     drs, brakesTemperatureRL, brakesTemperatureRR, brakesTemperatureFL, brakesTemperatureFR,
     tyresSurfaceTemperatureRL, tyresSurfaceTemperatureRR, tyresSurfaceTemperatureFL,
     tyresSurfaceTemperatureFR, engineTemperature FROM telemetryData""")

    masterTelemetryDf = pd.DataFrame(cur.fetchall())

    telemetryCols = ["frameIdentifier", "speed", "throttle", "steer", "brake", "clutch", "gear", "engineRPM",
    "drs", "brakesTemperatureRL", "brakesTemperatureRR", "brakesTemperatureFL", "brakesTemperatureFR",
    "tyresSurfaceTemperatureRL", "tyresSurfaceTemperatureRR",
    "tyresSurfaceTemperatureFL", "tyresSurfaceTemperatureFR", "engineTemperature"]

    masterTelemetryDf.columns = telemetryCols
    return masterTelemetryDf

def masterSetupData(connection):
    cur = connection.cursor()
    cur.execute("SELECT * FROM CarSetupData")
    masterSetupDf = pd.DataFrame(cur.fetchall())
    setupCols = [ "Index", "frameIdentifier", "SessionTime", "frontWing", "rearWing", "onThrottle", "offThrottle", "frontCamber",
    "rearCamber", "frontToe", "rearToe", "frontSuspension", "rearSuspension", "frontAntiRollBar",
    "rearAntiRollBar", "frontSuspensionHeight", "rearSuspensionHeight", "brakePressure", "brakeBias",
    "rearLeftTyrePressure", "rearRightTyrePressure", "frontLeftTyrePressure", "frontRightTyrePressure",
    "ballast","fuelLoad"]
    masterSetupDf.columns = setupCols
    masterSetupDf = masterSetupDf.drop("Index", 1)
    masterSetupDf = masterSetupDf.drop("SessionTime", 1)
    return masterSetupDf


def masterStatusData(connection):
    cur = connection.cursor()
    cur.execute("""SELECT frameIdentifier, fuelMix, FrontBrakeBias, fuelInTank, fuelRemainingLaps,
    tyresWearRL, tyresWearRR, tyresWearFL, tyresWearFR, actualTyreCompound, tyresAgeLaps,
    frontLeftWingDamage, frontRightWingDamage, rearWingDamage, gearBoxDamage, engineDamage
    FROM carStatusData""")
    masterStatusDf = pd.DataFrame(cur.fetchall())
    statusCols = ["frameIdentifier", "fuelMix", "FrontBrakeBias", "fuelInTank", "fuelRemainingLaps",
    "tyresWearRL", "tyresWearRR", "tyresWearFL", "tyresWearFR", "actualTyreCompound", "tyresAgeLaps",
    "frontLeftWingDamage", "frontRightWingDamage", "rearWingDamage", "gearBoxDamage", "engineDamage"]
    masterStatusDf.columns = statusCols
    return masterStatusDf

def masterSessionData(connection):
    cur = connection.cursor()
    cur.execute("""SELECT frameIdentifier, weather, trackTemperature, trackLength, trackId from SessionData""")
    sessionCols = ["frameIdentifier", "weather", "trackTemperature", "trackLength", "trackId"]
    masterSessionDf = pd.DataFrame(cur.fetchall())
    masterSessionDf.columns = sessionCols
    return masterSessionDf

def masterData(conn):

    masterLapDf = masterLapData(conn)
    masterPacketDf = masterPacketData(conn)
    masterMotionDf = masterMotionData(conn)
    masterTelemetryDf = masterTelemetryData(conn)
    masterSetupDf = masterSetupData(conn)
    masterStatusDf = masterStatusData(conn)
    masterSessionDf = masterSessionData(conn)

    masterDf = masterLapDf.merge(masterPacketDf, on="frameIdentifier")#.merge(masterMotionDf, on="frameIdentifier").merge(masterTelemetryDf, on="frameIdentifier").merge(masterSetupDF, on="frameIdentifier").merge(masterStatusDf, on="frameIdentifier")
    masterDf = masterDf.merge(masterMotionDf, on="frameIdentifier")
    masterDf = masterDf.merge(masterTelemetryDf, on="frameIdentifier")
    masterDf = masterDf.merge(masterStatusDf, on="frameIdentifier")
    masterSetupDf = masterSetupDf.merge(masterSessionDf, on="frameIdentifier")

    return masterDf, masterSetupDf

def sessionReducer(df):
    df = df.drop("Header", 1)
    df = df.drop("weatherForecastSamples", 1)
    return df

def masterDfToSQL(df1, df2, conn):
    df1.to_sql("MasterData", con = conn, schema = None, if_exists = "replace")
    df2.to_sql("MasterSetupData", con = conn, schema = None, if_exists = "replace")

def DBExpand(database):
    column_names = ["pkt_id", "timestamp", "packetFormat", "gameMajorVersion", "gameMinorVersion", "packetVersion", "packetId", "sessionUID", "sessionTime", "frameIdentifier", "playerCarIndex", "packet"]
    conn = connect(database)
    logging.info("Expanding " + str(database))
    df = selectALL(conn)
    df.columns = column_names
    df.reset_index(drop = True, inplace = True)
    motionDF, sessionDF, lapDataDF, eventDF, carSetupsDF, carTelemetryDF, carStatusDF = globalFormater(df)
    sessionDF = sessionReducer(sessionDF)
    Tables(motionDF, sessionDF, lapDataDF, eventDF, carSetupsDF, carTelemetryDF, carStatusDF, conn)
    masterDf, masterSetupDf = masterData(conn)
    masterDfToSQL(masterDf, masterSetupDf, conn)
    trainingCalculations(database)
    conn.close()
