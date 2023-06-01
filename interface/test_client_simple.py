#!/usr/bin/env python3
import time
import timeit
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import socket

#HOST = '192.168.0.102'  # The server's hostname or IP address
HOST = '192.168.134.41'  # The server's hostname or IP address
PORT = 1001        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))

    buf_len = 256*1024
    loops = 0
    with open("test", "wb") as file:
        while True:
            buffer = sock.recv(buf_len)
            bl = len(buffer)
            #print(buffer)
            #print(bl)
            elements = buffer.split(b"\x00\x00\x00\x00\x00")
            for b in elements:
                file.write(b)

            loops += 1
            if loops > 100:
                break

        """
        for b in elements:
            #print(b)
            try:
                # pass
                print(int.from_bytes(b, byteorder="little"))
            except ValueError:
                continue
        """
