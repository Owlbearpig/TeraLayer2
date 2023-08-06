import socket
import time
import binascii
from bitstring import BitArray
from numpy import pi
import matplotlib.pyplot as plt

HOST = "192.168.178.24"
PORT = 1001
c_ = 2 ** 6 * 2 * pi * 2 ** (-11)  # conversion factor


def flip(s):
    s_ = [s[i:i + 2] for i in range(0, len(s), 2)]

    return "".join(list(reversed(s_)))


with open("dump", "wb") as file:
    y = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        t0 = time.time()
        cntr = 0
        cntr1 = 0
        buf_len = 64
        chunk = 8
        cntr_sum = 0
        while True:
            # time.sleep(0.01)
            buffer = sock.recv(buf_len)
            # print(time.time() - t0)
            print(buffer)
            print(len(buffer))
            file.write(buffer)
            for j in range(len(buffer) // chunk):
                hexdata = binascii.hexlify(buffer[j * chunk:(j + 1) * chunk]).decode()
                print(hexdata)
                s = [hexdata[i:i + 2] for i in range(0, len(hexdata), 2)]
                # print(s)
                input_str = '0x' + "".join(list(reversed(s)))
                # print(input_str)
                clk_cycle_cnt = int(input_str, 16)
                # print(clk_cycle_cnt, clk_cycle_cnt/)

                c = BitArray(hex=input_str)
                c_bin = str(c.bin)

                fpga_cntr, d0, d1, d2 = c_bin[-64:-48], c_bin[-48:-32], c_bin[-32:-16], c_bin[-16:]
                # print(cntr, fpga_cntr, int(fpga_cntr, 2))
                d0, d1, d2 = c_ * int(d0, 2), c_ * int(d1, 2), c_ * int(d2, 2)
                fpga_cntr = int(fpga_cntr, 2)

                # print(fpga_cntr, d0, d1, d2)
                cntr += 1
                cntr_sum += fpga_cntr
                y.append(fpga_cntr)
                if cntr > 1000:
                    pass
                    # plt.plot(y)
                    # plt.show()
                if (cntr % 100) == 0:
                    print("cntr", cntr)
                    print("rate", (time.time() - t0) / cntr)
                    # cntr = 0
                    print("cntr_sum", cntr_sum)
                    # cntr_sum = 0
                    # print(time.time() - t0)
