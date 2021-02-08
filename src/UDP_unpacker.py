import socket
import enum
import ctypes
from src.packedLittleEndian import PackedLittleEndianStructure
from src.datatypes import *

def unpackUDPpacket(packet: bytes):
    """unpacks the UDP packet from binary format to corresponding class in datatypes.
    Also used to unpack blob data from sql table
    """
    actual_packet_size = len(packet)
    header = PacketHeader.from_buffer_copy(packet)
    key = (header.packetFormat, header.packetVersion, header.packetId)
    packet_type = HeaderFieldsToPacketType[key]
    header_size = ctypes.sizeof(PacketHeader)

    if actual_packet_size < header_size:
        raise UnpackError(
            f"Bad telemetry packet: too short ({actual_packet_size} bytes)."
        )

    header = PacketHeader.from_buffer_copy(packet)
    key = (header.packetFormat, header.packetVersion, header.packetId)

    if key not in HeaderFieldsToPacketType:
        raise UnpackError(
            f"Bad telemetry packet: no match for key fields {key!r}."
        )

    packet_type = HeaderFieldsToPacketType[key]

    expected_packet_size = ctypes.sizeof(packet_type)

    if actual_packet_size != expected_packet_size:
        raise UnpackError(
            "Bad telemetry packet: bad size for {} packet; expected {} bytes but received {} bytes.".format(
                packet_type.__name__, expected_packet_size, actual_packet_size
            )
        )
    return packet_type.from_buffer_copy(packet)

class UnpackError(Exception):
    """Exception for packets that cannot be unpacked"""
