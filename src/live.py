import threading
import numpy as np
import pandas as pd
from src.UDP_unpacker import unpackUDPpacket
from src.databaseUnpacker import localFormat
import time

class live_storage(object):
    '''stores live data in memory'''
    def __init__(self):
        self.packet = None
        self.motionData = []
        self.sessionData = []
        self.lapData = []
        self.setupData = []
        self.telemetryData = []
        self.statusData = []
        self.type = None
    def accept_packet(self, packet):
        type_to_function = {0:"motion", 1:"session", 2:"lap", 3:"event", 4:"participants",
        5:"setup", 6:"telemetry", 7:"status", 8:"finalClassification", 9:"lobbyInfo"}
        self.packet = unpackUDPpacket(packet)
        self.type = self.packet.header.packetId
        getattr(live_storage, type_to_function[self.type])(self)
    def motion(self):
        self.motionData.append(localFormat(self.packet, self.type).arr)
    def session(self):
        self.sessionData.append(localFormat(self.packet, self.type).arr)
    def lap(self):
        self.lapData.append(localFormat(self.packet, self.type).arr)
    def event(self):
        pass
    def participants(self):
        pass
    def setup(self):
        self.setupData.append(localFormat(self.packet, self.type).arr)
    def telemetry(self):
        self.telemetryData.append(localFormat(self.packet, self.type).arr)
    def status(self):
        self.statusData.append(localFormat(self.packet, self.type).arr)
    def finalClassification(self):
        pass
    def lobbyInfo(self):
        pass
    def getMotion(self):
        return self.motionData
    def getSession(self):
        return self.sessionData
    def getLap(self):
        return self.lapData
    def getSetup(self):
        return self.setupData
    def getTelemetry(self):
        return self.telemetryData
    def getStatus(self):
        return self.statusData


class liveMerged(threading.Thread):
    def __init__(self, main_Data, q, DONE):
        super().__init__(name = "live_merge")
        self.motion = pd.DataFrame()
        self.session = pd.DataFrame()
        self.lap = pd.DataFrame()
        self.setup = pd.DataFrame()
        self.telemetry = pd.DataFrame()
        self.status = pd.DataFrame()
        self.previousIndex = [0]*6
        self.main = pd.DataFrame()
        self.q = q
        self.DONE = DONE
        self.set = False
        self.quitflag = False
        self.mainData = main_Data

        self.motionCols = ["frameIdentifier", "SessionTime", "worldPositionX", "worldPositionY", "worldPositionZ", "worldVelocityX", "worldVelocityY",
        "worldVelocityZ","worldForwardDirX", "worldForwardDirY", "worldForwardDirZ", "worldRightDirX", "worldRightDirY",
        "worldRightDirZ","gForceLateral", "gForceLongitudinal","gForceVertical", "yaw", "pitch", "roll", "suspensionPositionRL",
        "suspensionPositionRR", "suspensionPositionFL", "suspensionPositionFR", "suspensionVelocityRL",
        "suspensionVelocityRR", "suspensionVelocityFL", "suspensionVelocityFR", "suspensionAccelerationRL", "suspensionAccelerationRR",
        "suspensionAccelerationFL", "suspensionAccelerationFR", "wheelSpeedRL", "wheelSpeedRR", "wheelSpeedFL", "wheelSpeedFR", "wheelSlipRL",
        "wheelSlipRR", "wheelSlipFL","wheelSlipFR", "localVelocityX", "localVelocityY", "localVelocityZ", "angularVelocityX", "angularVelocityY",
        "angularVelocityZ", "angularAccelerationX", "angularAccelerationY", "angularAccelerationZ", "frontWheelsAngle"]

        self.sessionCols = ["frameIdentifier", "SessionTime", "Header", "weather", "trackTemperature", "airTemperature",
        "totalLaps", "trackLength", "sessionType", "trackId", "formula", "sessionTimeLeft",
        "sessionDuration", "pitSpeedLimit","gamePaused", "isSpectating", "spectatorCarIndex",
        "sliProNativeSupport", "numMarshalZones", "marshalZones", "safetyCarStatus",
        "networkGame", "numWeatherForecastSamples", "weatherForecastSamples"]

        self.lapDataCols = ["frameIdentifier", "SessionTime","lastLapTime", "currentLapTime", "sector1TimeInMS", "sector2TimeInMS", "bestLapTime",
        "bestLapNum", "bestLapSector1TimeInMS", "bestLapSector2TimeInMS", "bestLapSector3TimeInMS",
        "bestOverallSector1TimeInMS", "bestOverallSector1LapNum", "bestOverallSector2TimeInMS",
        "bestOverallSector2LapNum", "bestOverallSector3TimeInMS", "bestOverallSector3LapNum",
        "lapDistance", "totalDistance", "safetyCarDelta", "carPosition", "currentLapNum",
        "pitStatus", "sector", "currentLapInvalid", "penalties", "gridPosition", "driverStatus",
        "resultStatus"]

        self.carSetupsCols = ["frameIdentifier", "SessionTime", "frontWing", "rearWing", "onThrottle", "offThrottle", "frontCamber",
        "rearCamber", "frontToe", "rearToe", "frontSuspension", "rearSuspension", "frontAntiRollBar",
        "rearAntiRollBar", "frontSuspensionHeight", "rearSuspensionHeight", "brakePressure", "brakeBias",
        "rearLeftTyrePressure", "rearRightTyrePressure", "frontLeftTyrePressure", "frontRightTyrePressure",
        "ballast","fuelLoad"]

        self.carTelemetryCols = ["frameIdentifier", "SessionTime", "speed", "throttle", "steer", "brake", "clutch", "gear", "engineRPM",
        "drs", "revLightsPercent", "brakesTemperatureRL", "brakesTemperatureRR",
        "brakesTemperatureFL", "brakesTemperatureFR", "tyresSurfaceTemperatureRL", "tyresSurfaceTemperatureRR",
        "tyresSurfaceTemperatureFL", "tyresSurfaceTemperatureFR", "tyresInnerTemperatureRL",
        "tyresInnerTemperatureRR", "tyresInnerTemperatureFL", "tyresInnerTemperatureFR",
        "engineTemperature", "tyresPressureRL", "tyresPressureRR", "tyresPressureFL", "tyresPressureFR",
        "surfaceTypeRL", "surfaceTypeRR", "surfaceTypeFL", "surfaceTypeFR"]

        self.carStatusCols = ["frameIdentifier", "SessionTime", "tractionControl", "antiLockBrakes", "fuelMix", "FrontBrakeBias",
        "pitLimiterStatus", "fuelInTank", "fuelCapacity", "fuelRemainingLaps", "maxRPM", "idleRPM", "maxGears",
        "drsAllowed", "drsActivationDistance", "tyresWearRL", "tyresWearRR", "tyresWearFL", "tyresWearFR",
        "actualTyreCompound", "visualTyreCompound", "tyresAgeLaps", "tyresDamageRL", "tyresDamageRR",
        "tyresDamageFL", "tyresDamageFR", "frontLeftWingDamage", "frontRightWingDamage", "rearWingDamage",
        "drsFault", "engineDamage", "gearBoxDamage", "vehicleFiaFlags", "ersStoreEnergy", "ersDeployMode",
        "ersHarvestedThisLapMGUK", "ersHarvestedThisLapMGUH", "ersDeployedThisLap"]

        self.masterMotion = ["frameIdentifier", "worldPositionX", "worldPositionY", "worldPositionZ",
            "worldVelocityX", "worldVelocityY", "worldVelocityZ", "yaw", "pitch", "roll"]

        self.masterSession = ["frameIdentifier", "weather", "trackTemperature", "trackLength", "trackId"]

        self.masterLap = ["frameIdentifier", "lastLapTime", "currentLapTime", "currentLapNum", "lapDistance", "carPosition", "sector"]

        self.masterSetup = ["frameIdentifier", "SessionTime", "frontWing", "rearWing", "onThrottle", "offThrottle", "frontCamber",
        "rearCamber", "frontToe", "rearToe", "frontSuspension", "rearSuspension", "frontAntiRollBar",
        "rearAntiRollBar", "frontSuspensionHeight", "rearSuspensionHeight", "brakePressure", "brakeBias",
        "rearLeftTyrePressure", "rearRightTyrePressure", "frontLeftTyrePressure", "frontRightTyrePressure",
        "ballast","fuelLoad"]

        self.masterTelemetry = ["frameIdentifier", "speed", "throttle", "steer", "brake", "clutch", "gear", "engineRPM",
        "drs", "brakesTemperatureRL", "brakesTemperatureRR", "brakesTemperatureFL", "brakesTemperatureFR",
        "tyresSurfaceTemperatureRL", "tyresSurfaceTemperatureRR",
        "tyresSurfaceTemperatureFL", "tyresSurfaceTemperatureFR", "engineTemperature"]

        self.masterStatus = ["frameIdentifier", "fuelMix", "FrontBrakeBias", "fuelInTank", "fuelRemainingLaps",
        "tyresWearRL", "tyresWearRR", "tyresWearFL", "tyresWearFR", "actualTyreCompound", "tyresAgeLaps",
        "frontLeftWingDamage", "frontRightWingDamage", "rearWingDamage", "gearBoxDamage", "engineDamage"]


        self.finalCols = None


        self.final = pd.DataFrame()


    def getData(self):
        main_data = self.mainData
        motion = list(main_data.getMotion())
        session = list(main_data.getSession())
        lap = list(main_data.getLap())
        setup = list(main_data.getSetup())
        telemetry = list(main_data.getTelemetry())
        status = list(main_data.getStatus())

        if motion:
            self.motion = pd.DataFrame(motion[self.previousIndex[0]+1:], columns = self.motionCols)
            self.session = pd.DataFrame(session[self.previousIndex[1]+1:], columns = self.sessionCols)
            self.lap = pd.DataFrame(lap[self.previousIndex[2]+1:], columns = self.lapDataCols)
            self.setup = pd.DataFrame(setup[self.previousIndex[3]+1:], columns = self.carSetupsCols)
            self.telemetry = pd.DataFrame(telemetry[self.previousIndex[4]+1:], columns = self.carTelemetryCols)
            self.status = pd.DataFrame(status[self.previousIndex[5]+1:], columns = self.carStatusCols)

            self.motion = self.motion[self.masterMotion]
            self.session = self.session[self.masterSession]
            self.lap = self.lap[self.masterLap]
            self.setup = self.setup[self.masterSetup]
            self.telemetry = self.telemetry[self.masterTelemetry]
            self.status = self.status[self.masterStatus]

            self.previousIndex[0] = self.previousIndex[0] + len(self.motion.index)
            self.previousIndex[1] = self.previousIndex[1] + len(self.session.index)
            self.previousIndex[2] = self.previousIndex[2] + len(self.lap.index)
            self.previousIndex[3] = self.previousIndex[3] + len(self.setup.index)
            self.previousIndex[4] = self.previousIndex[4] + len(self.telemetry.index)
            self.previousIndex[5] = self.previousIndex[5] + len(self.status.index)

    def merge(self):
        if not self.motion.empty:
            self.main = self.lap.merge(self.motion, on="frameIdentifier")
            self.main = self.main.merge(self.telemetry, on="frameIdentifier")
            self.main = self.main.merge(self.status, on="frameIdentifier")
            if self.set == False:
                self.finalColumns = self.main.columns
                self.final = pd.DataFrame(columns = self.finalColumns)
                self.set = True

    def concat(self):
        if not self.main.empty:
            self.final = pd.concat([self.final, self.main]).reset_index(drop=True)
            self.final.set_index('frameIdentifier', inplace=True)

    def isChanged(self):
        """resets the dataframe every lap"""
        if not self.final.empty:
            isLapChanged = (self.final["currentLapNum"].shift(1, fill_value=self.final["currentLapNum"].head(1)) != self.final["currentLapNum"]).to_numpy()
            if True in isLapChanged:
                self.final = pd.DataFrame(columns = self.finalColumns)

    def run(self):
        while not self.quitflag:
            self.getData()
            self.merge()
            self.concat()
            self.isChanged()
            if not self.final.empty:
                self.q.put(self.final)

        self.q.put(self.DONE)
        
    def requestQuit(self):
        self.quitflag = True
