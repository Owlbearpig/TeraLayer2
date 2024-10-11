import socket
import time
import numpy as np
import binascii
from numpy import pi
import matplotlib.pyplot as plt
import os
# from mpl_settings import *

if "nt" in os.name.lower():
    HOST = "192.168.178.24"
else:
    HOST = "192.168.134.69"

PORT = 1001
c_ = 2 ** 6 * 2 * pi * 2 ** (-11)  # conversion factor
t0 = time.time()
print(c_)

def format_recv_data(buffer_):
    width_tdata = 8  # width_tdata 64 bit / 8 (bit / byte) = 8 byte
    # concat = [4, 2, 2]  # byte
    concat = [2, 2, 2, 2]  # byte
    t_ = []

    slice0_, slice1_, slice2_, slice3_ = [], [], [], []  # number of lists should be == len(concat)
    for j in range(len(buffer_) // width_tdata):
        # split into parts of length given by width_tdata in ram_writer core
        tdata = binascii.hexlify(buffer_[j * width_tdata:(j + 1) * width_tdata]).decode()

        # split into bytes and reverse
        bytes_ = [tdata[i:i + 2] for i in reversed(range(0, len(tdata), 2))]
        # print(bytes_)

        # split into slices of length given by concat(2) core and convert to dec
        slice_vals = []
        for i, slice_width in enumerate(concat):
            idx0 = sum(concat[:i])
            slice_ = "".join(bytes_[idx0:idx0 + slice_width])
            val = int(slice_, 16)
            slice_vals.append(val)

        t_.append(time.time() - t0)
        slice0_.append(slice_vals[0])
        slice1_.append(c_ * slice_vals[-1])
        slice2_.append(c_ * slice_vals[-2])
        slice3_.append(c_ * slice_vals[-3])

    return t_, slice0_, slice1_, slice2_, slice3_


with open("dump", "wb") as file:
    y, d0_, d1_, d2_ = [], [], [], []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        cntr = 0
        buf_len = 128

        t0 = time.time()
        while True:
            buffer = sock.recv(buf_len)
            #print(f"Received {len(buffer)} bytes")
            file.write(buffer)
            t, fpga_cntr, d0, d1, d2 = format_recv_data(buffer)
            print(fpga_cntr)

            y.extend(fpga_cntr)
            d0_.extend(d0)
            d1_.extend(d1)
            d2_.extend(d2)

            if cntr > 7e3:
                cntr = 0
                fig, (ax0, ax1) = plt.subplots(2, 1)
                ax0.plot(y)
                ax1.plot(d0_)
                ax1.plot(d1_)
                ax1.plot(d2_)
                print(np.mean(d0_), np.std(d0_))
                print(np.mean(d1_), np.std(d1_))
                print(np.mean(d2_), np.std(d2_))
                plt.show()

            dt = time.time() - t0
            try:
                print(f"Avg. T: {dt / cntr}, (cnts: {y[-1]})\n")
            except ZeroDivisionError:
                pass

            cntr += len(fpga_cntr)
