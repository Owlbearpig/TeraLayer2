import socket
import time
import numpy as np
import binascii
import matplotlib.animation as animation
from numpy import pi
import matplotlib.pyplot as plt
import os
from mpl_settings import *

if "nt" in os.name.lower():
    HOST = "192.168.178.24"
else:
    HOST = "192.168.134.69"

PORT = 1001
c_ = 2 ** 6 * 2 * pi * 2 ** (-11)  # conversion factor
t0 = time.time()

round = np.round


def format_recv_data(buffer_):
    width_tdata = 8  # width_tdata 64 bit / 8 (bit / byte) = 8 byte
    # concat = [4, 2, 2]  # byte
    concat = [2, 2, 2, 2]  # byte
    t = []

    slice0_, slice1_, slice2_, slice3_ = [], [], [], []  # number of lists should be == len(concat)
    for j in range(len(buffer_) // width_tdata):
        # split into parts of length given by width_tdata in ram_writer core
        tdata = binascii.hexlify(buffer_[j * width_tdata:(j + 1) * width_tdata]).decode()

        # split into bytes and reverse
        bytes_ = [tdata[i:i + 2] for i in reversed(range(0, len(tdata), 2))]

        # split into slices of length given by concat(2) core and convert to dec
        slice_vals = []
        for i, slice_width in enumerate(concat):
            idx0 = sum(concat[:i])
            slice_ = "".join(bytes_[idx0:idx0 + slice_width])
            val = int(slice_, 16)
            slice_vals.append(val)

        t.append(time.time() - t0)
        slice0_.append(slice_vals[0])
        slice1_.append(c_ * slice_vals[-1])
        slice2_.append(c_ * slice_vals[-2])
        slice3_.append(c_ * slice_vals[-3])

    return t, slice0_, slice1_, slice2_, slice3_


with open("dump", "wb") as file:
    y, d0_, d1_, d2_ = [], [], [], []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        buf_len = 128

        # Create figure for plotting
        f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.xaxis.tick_top()
        ax.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        # txt = ax.text(.20, .5, "here", fontsize=15)

        # Format plot
        ax.set_ylim(600, 700)
        ax2.set_ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.20)

        ax2.set_xlabel('Measurement counter')
        ax.set_ylabel("Layer width (µm)")
        ax.yaxis.set_label_coords(-0.1, -0.0)

        cntr = []
        plot_buf_len = 8
        line_d0, = ax2.plot(16 * plot_buf_len * [1])
        line_d1, = ax.plot(16 * plot_buf_len * [1])
        line_d2, = ax2.plot(16 * plot_buf_len * [1])

        mean_line_d0 = ax2.axhline(y=0, color='r', linestyle='-')
        mean_line_d1 = ax.axhline(y=0, color='r', linestyle='-')
        mean_line_d2 = ax2.axhline(y=0, color='r', linestyle='-')

        xs, d0, d1, d2 = [], [], [], []
        loop_cntr = 0
        while True:
            # time.sleep(0.001)
            buffer = sock.recv(buf_len)
            file.write(buffer)
            resp = format_recv_data(buffer)

            xs.extend(resp[1])
            d0_, d1_, d2_ = resp[-3:]
            d0.extend(resp[2]), d1.extend(resp[3]), d2.extend(resp[4])

            if loop_cntr == plot_buf_len-1:
                line_d0.set_ydata(d0)
                line_d1.set_ydata(d1)
                line_d2.set_ydata(d2)

                # tick at every 10 points
                ax.set_xticks(range(0, 16*plot_buf_len+10, 10), range(xs[0], xs[-1]+10, 10))

                dt = 1000 * (time.time() - t0) / xs[-1]
                mean_d2, mean_d1, mean_d0 = np.mean(d2_), np.mean(d1_), np.mean(d0_)
                avg = f"Mean: {round(mean_d2, 1)}, {round(mean_d1, 1)}, {round(mean_d0, 1)} (µm)\n"
                std = f"Std: {round(np.std(d2_), 1)}, {round(np.std(d1_), 1)}, {round(np.std(d0_), 1)} (µm)"

                mean_line_d0.set_ydata(mean_d0)
                mean_line_d1.set_ydata(mean_d1)
                mean_line_d2.set_ydata(mean_d2)

                ax.set_title(f'Period: {round(dt, 2)} ms, {round(1000/dt, 0)} Hz\n' + avg + std)
                plt.pause(0.0004)

                xs, d0, d1, d2 = [], [], [], []
                loop_cntr = 0
            else:
                loop_cntr += 1
