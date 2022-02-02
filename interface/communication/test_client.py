#!/usr/bin/env python3
import time
import timeit
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import socket

MB = 10**6

#HOST = '192.168.0.102'  # The server's hostname or IP address
HOST = '192.168.178.24'  # The server's hostname or IP address
PORT = 1001        # The port used by the server


total_byte_cnt, avg_byte_cnt = 0, 0
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))

    chunk_len = 16*10
    bin0, bin1, bin2, bin3 = [], [], [], []
    data = []
    t0 = timeit.default_timer()
    while True:
        buffer = sock.recv(chunk_len)
        for i in range(0, len(buffer), chunk_len):
            slice_ = buffer[i:(i+chunk_len)]
            slice0, slice1, slice2, slice3 = slice_[0:4], slice_[4:8], slice_[8:12], slice_[12:16]
            bin0.append(int.from_bytes(slice0, byteorder='little', signed=True))
            bin1.append(int.from_bytes(slice1, byteorder='little', signed=True))
            bin2.append(int.from_bytes(slice2, byteorder='little', signed=True))
            bin3.append(int.from_bytes(slice3, byteorder='little', signed=True))

            print(slice0.hex(), slice1.hex(), slice2.hex(), slice3.hex())

        bl = len(buffer) // 2  # 4 bit / hex

        total_byte_cnt += bl
        avg_byte_cnt += bl
        dt = timeit.default_timer() - t0
        if dt >= 2:
            print(f"{(avg_byte_cnt / MB) / dt} MB/s. {total_byte_cnt} total bytes received")
            avg_byte_cnt = 0
            t0 = timeit.default_timer()

        if total_byte_cnt >= 4*256*1024:
            break


def fft(data):
    Y = np.fft.rfft(data - np.mean(data))

    sample_rate = 25000
    freq = np.fft.rfftfreq(data.size, d=1. / sample_rate)

    return freq, Y


bins = [np.array(bin) for bin in [bin0, bin1, bin2, bin3]]

plt.plot(*fft(bins[0]), label='bin0 fft')
#plt.plot(*fft(bins[1]), label='bin1 fft')
plt.plot(*fft(bins[2]), label='bin2 fft')
#plt.plot(*fft(bins[3]), label='bin3 fft')
plt.legend()
plt.show()

plt.plot(bins[0], label='bin0')
#plt.plot(bins[1], label='bin1')
plt.plot(bins[2], label='bin2')
#plt.plot(bins[3], label='bin3')
plt.legend()
plt.show()
