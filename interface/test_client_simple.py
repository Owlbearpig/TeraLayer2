#!/usr/bin/env python3
import time
import timeit
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import socket

#HOST = '192.168.0.102'  # The server's hostname or IP address
#HOST = '192.168.134.41'  # The server's hostname or IP address
HOST = '192.168.178.53'  # The server's hostname or IP address
PORT = 1001        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    while True:
        buf_len = 8
        buffer = sock.recv(buf_len)
        print(buffer)
