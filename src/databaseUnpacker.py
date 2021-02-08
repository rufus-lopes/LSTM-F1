import numpy as np
import pandas as pd
from src.datatypes import *
import ctypes

class localFormat(object):
    """Takes a packet and packetType as input, and returns data formatted in a list ready
    for adding to pandas DataFrame"""

    def __init__(self, packet, packetType):
        localFormatter = {0:"motion", 1:"session", 2:"lap", 3:"event", 4:"participants",
                5:"carSetups", 6:"carTelemetry", 7:"carStatus", 8:"finalClassification", 9:"lobbyInfo"}
        self.packet = packet
        self.df = pd.DataFrame(None)
        self.arr = getattr(localFormat, localFormatter[packetType])(self)
    def motion(self):
        motion_packet = self.packet
        player_motion = motion_packet.carMotionData[motion_packet.header.playerCarIndex]
        arr = np.ctypeslib.as_array(player_motion).tolist()
        arr = list(arr)

        sessTime = motion_packet.header.sessionTime
        arr.insert(0, sessTime)
        frame = self.packet.header.frameIdentifier
        arr.insert(0, frame)
        #getting additional data for player car only
        suspensionPositionRL = motion_packet.suspensionPosition[0]
        suspensionPositionRR = motion_packet.suspensionPosition[1]
        suspensionPositionFL = motion_packet.suspensionPosition[2]
        suspensionPositionFR = motion_packet.suspensionPosition[3]
        suspensionVelocityRL = motion_packet.suspensionVelocity[0]
        suspensionVelocityRR = motion_packet.suspensionVelocity[1]
        suspensionVelocityFL = motion_packet.suspensionVelocity[2]
        suspensionVelocityFR = motion_packet.suspensionVelocity[3]
        suspensionAccelerationRL = motion_packet.suspensionAcceleration[0]
        suspensionAccelerationRR = motion_packet.suspensionAcceleration[1]
        suspensionAccelerationFL = motion_packet.suspensionAcceleration[2]
        suspensionAccelerationFR = motion_packet.suspensionAcceleration[3]
        wheelSpeedRL = motion_packet.wheelSpeed[0]
        wheelSpeedRR = motion_packet.wheelSpeed[1]
        wheelSpeedFL = motion_packet.wheelSpeed[2]
        wheelSpeedFR = motion_packet.wheelSpeed[3]
        wheelSlipRL = motion_packet.wheelSlip[0]
        wheelSlipRR = motion_packet.wheelSlip[1]
        wheelSlipFL = motion_packet.wheelSlip[2]
        wheelSlipFR = motion_packet.wheelSlip[3]
        localVelocityX = motion_packet.localVelocityX
        localVelocityY = motion_packet.localVelocityY
        localVelocityZ = motion_packet.localVelocityZ
        angularVelocityX = motion_packet.angularVelocityX
        angularVelocityY = motion_packet.angularVelocityY
        angularVelocityZ = motion_packet.angularVelocityZ
        angularAccelerationX = motion_packet.angularAccelerationX
        angularAccelerationY = motion_packet.angularAccelerationY
        angularAccelerationZ = motion_packet.angularAccelerationZ
        frontWheelsAngle = motion_packet.frontWheelsAngle
        playerData = [suspensionPositionRL, suspensionPositionRR, suspensionPositionFL, suspensionPositionFR, suspensionVelocityRL,
                     suspensionVelocityRR, suspensionVelocityFL, suspensionVelocityFR, suspensionAccelerationRL, suspensionAccelerationRR,
                     suspensionAccelerationFL, suspensionAccelerationFR, wheelSpeedRL, wheelSpeedRR, wheelSpeedFL, wheelSpeedFR, wheelSlipRL,
                     wheelSlipRR, wheelSlipFL, wheelSlipFR, localVelocityX, localVelocityY, localVelocityZ, angularVelocityX, angularVelocityY,
                     angularVelocityZ, angularAccelerationX, angularAccelerationY, angularAccelerationZ, frontWheelsAngle]
        arr += playerData
        return arr
    def session(self):
        arr = np.ctypeslib.as_array(self.packet).tolist()
        arr = list(arr)
        sessTime = self.packet.header.sessionTime
        arr.insert(0, sessTime)
        frame = self.packet.header.frameIdentifier
        arr.insert(0, frame)
        return arr
    def lap(self):
        arr = np.ctypeslib.as_array(self.packet.lapData[self.packet.header.playerCarIndex]).tolist()
        arr = list(arr)
        sessTime = self.packet.header.sessionTime
        arr.insert(0, sessTime)
        frame = self.packet.header.frameIdentifier
        arr.insert(0, frame)
        return arr
    def event(self):
        arr = np.ctypeslib.as_array(self.packet.eventStringCode)#.tolist()
        arr = [arr]
        sessTime = self.packet.header.sessionTime
        arr.insert(0, sessTime)
        frame = self.packet.header.frameIdentifier
        arr.insert(0, frame)
        return arr
    def participants(self):
        arr = np.ctypeslib.as_array(self.packet.participants[self.packet.header.playerCarIndex]).tolist()
        arr = list(arr)
        sessTime = self.packet.header.sessionTime
        arr.insert(0, sessTime)
        frame = self.packet.header.frameIdentifier
        arr.insert(0, frame)
        return arr
    def carSetups(self):
        arr = np.ctypeslib.as_array(self.packet.carSetups[self.packet.header.playerCarIndex]).tolist()
        arr = list(arr)
        sessTime = self.packet.header.sessionTime
        arr.insert(0, sessTime)
        frame = self.packet.header.frameIdentifier
        arr.insert(0, frame)
        return arr
    def carTelemetry(self):

        arr = np.ctypeslib.as_array(self.packet.carTelemetryData[self.packet.header.playerCarIndex]).tolist()
        arr = list(arr)
        sessTime = self.packet.header.sessionTime
        arr.insert(0, sessTime)

        #expanding out arrays within packet array.
        #maybe a better way to do this
        flat = []
        for i in arr:
            if isinstance(i, np.ndarray):
                for j in i:
                    flat.append(j)
            else:
                flat.append(i)
        arr = flat
        frame = self.packet.header.frameIdentifier
        arr.insert(0, frame)
        return arr
    def carStatus(self):
        arr = np.ctypeslib.as_array(self.packet.carStatusData[self.packet.header.playerCarIndex]).tolist()
        arr = list(arr)
        sessTime = self.packet.header.sessionTime
        arr.insert(0, sessTime)
        flat = []
        for i in arr:
            if isinstance(i, np.ndarray):
                for j in i:
                    flat.append(j)
            else:
                flat.append(i)
        arr = flat
        frame = self.packet.header.frameIdentifier
        arr.insert(0, frame)
        return arr
    def finalClassification(self):
        arr = np.ctypeslib.as_array(self.packet).tolist()
        arr = list(arr)
        sessTime = self.packet.header.sessionTime
        arr.insert(0, sessTime)
        frame = self.packet.header.frameIdentifier
        arr.insert(0, frame)
        return arr
    def lobbyInfo(self):
        arr = np.ctypeslib.as_array(self.packet).tolist()
        arr = list(arr)
        sessTime = self.packet.header.sessionTime
        arr.insert(0, sessTime)
        frame = self.packet.header.frameIdentifier
        arr.insert(0, frame)
        return arr
