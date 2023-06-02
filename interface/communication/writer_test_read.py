#!/usr/bin/env python3
import time
import timeit
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import socket

MB = 10**6
#plt.ion()
#fig = plt.figure()
#HOST = '192.168.0.102'  # The server's hostname or IP address
HOST = '192.168.178.53'  # The server's hostname or IP address
PORT = 1001        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))

    buf_len = 1024
    byte_cnt = 0
    t0 = timeit.default_timer()
    cntr = 0
    while True:
        buffer = sock.recv(buf_len)
        print(int.from_bytes(buffer, byteorder='little', signed=True))

        time.sleep(1)

