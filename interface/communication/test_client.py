#!/usr/bin/env python3

SETUP_MASK = 0xF
SETUP_SAWTOOTH = 1
SETUP_LOCKIN = 2
SETUP_SAMPLERATE = 3
SETUP_PDM_DAC = 4
SAWTOOTH_OFFSET = 1 << 4
SAWTOOTH_INCREMENT = 1 << 5
VAL0_MASK = 0xFF << 8
VAL1_MASK = 0xFFFF << 16
VAL10_MASK = 0xFFFFFF << 8

import socket

def rawADC2Volt(raw_val):
    return raw_val*(30.0+4.99)/(4.99*0xFFF)


HOST = '192.168.178.24'  # The server's hostname or IP address
PORT = 1001        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    #s.sendall(b'Hello, world')
    while True:
        buffer = s.recv(1024)
        print(buffer)
        #val0 = int((buffer & VAL0_MASK) >> 8)
        #val1 = int((buffer & VAL1_MASK) >> 16)
        #val10 = int((buffer & VAL10_MASK) >> 8)