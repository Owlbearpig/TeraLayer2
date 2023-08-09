import socket
import time
import numpy as np
import binascii
from bitstring import BitArray
from numpy import pi
import matplotlib.pyplot as plt
import os

if "nt" in os.name.lower():
    HOST = "192.168.178.24"
else:
    HOST = "192.168.134.69"

PORT = 1001
c_ = 2 ** 6 * 2 * pi * 2 ** (-11)  # conversion factor


def flip(s):
    s_ = [s[i:i + 2] for i in range(0, len(s), 2)]

    return "".join(list(reversed(s_)))


with open("dump", "wb") as file:
    y, d0_, d1_, d2_ = [], [], [], []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        cntr = 0
        buf_len = 128
        chunk = 8

        t0 = time.time()
        while True:
            # time.sleep(0.01)
            buffer = sock.recv(buf_len)
            #print(buffer)
            #print(f"Received {len(buffer)} bytes")
            file.write(buffer)
            for j in range(len(buffer) // chunk):
                hexdata = binascii.hexlify(buffer[j * chunk:(j + 1) * chunk]).decode()
                s = ["0x" + hexdata[i + 2:i + 4] + hexdata[i:i + 2] for i in range(0, len(hexdata), 4)]
                fpga_cntr = int(s[-1], 16)
                print(fpga_cntr, s)

                d0, d1, d2 = c_ * int(s[0], 16), c_ * int(s[1], 16), c_ * int(s[2], 16)
                print(d0, d1, d2)

                y.append(fpga_cntr)
                d0_.append(d0)
                d1_.append(d1)
                d2_.append(d2)

                if len(y) == 10000:
                    fig, (ax0, ax1) = plt.subplots(2, 1)
                    ax0.plot(y)
                    ax1.plot(d0_)
                    ax1.plot(d1_)
                    ax1.plot(d2_)
                    print(np.mean(d0_[:100]), np.std(d0_[:100]))
                    print(np.mean(d1_[:100]), np.std(d1_[:100]))
                    print(np.mean(d2_[:100]), np.std(d2_[:100]))
                    plt.show()

                dt = time.time() - t0
                try:
                    print(f"Avg. T: {dt / cntr}\n")
                except ZeroDivisionError:
                    pass
                cntr += 1
