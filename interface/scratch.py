from bitstring import BitArray
import binascii
from numpy import pi
import matplotlib.pyplot as plt
import socket
import matplotlib.animation as animation
import time
import datetime as dt

HOST = '192.168.178.53'  # HOME network
HOST = '192.168.134.69'  # UNI

PORT = 1001
c_ = 2 ** 6 * 2 * pi * 2 ** (-11)  # conversion factor

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    t0 = time.time()

    def pull_data():
        buf_len = 8
        buffer = sock.recv(buf_len)
        # print(buffer)

        hexdata = binascii.hexlify(buffer)
        s_ = hexdata.decode()
        # print(s_)
        s = [s_[i:i + 2] for i in range(0, len(s_), 2)]
        input_str = '0x' + "".join(list(reversed(s)))
        # print(input_str)
        c = BitArray(hex=input_str)
        c_bin = str(c.bin)
        #print(c_bin)
        d0, d1, d2 = c_bin[-45:-30], c_bin[-30:-15], c_bin[-15:]
        d0, d1, d2 = c_ * int(d0, 2), c_ * int(d1, 2), c_ * int(d2, 2)
        # print(d0, d1, d2)

        return d0, d1, d2

        # t = time.time()
        # return t, d0, d1, d2


    # Create figure for plotting
    fig, ax = plt.subplots()
    xs = []
    d0_, d1_, d2_ = [], [], []

    def animate(i, xs: list, d0_: list, d1_: list, d2_: list):
        # grab the data from thingspeak.com
        resp = pull_data()
        # Add x and y to lists
        xs.append(time.time() - t0)
        d0_.append(resp[0]), d1_.append(resp[1]), d2_.append(resp[2])

        # Limit x and y lists to 10 items
        xs = xs[-10:]
        d0_ = d0_[-10:]
        d1_ = d1_[-10:]
        d2_ = d2_[-10:]
        # Draw x and y lists
        ax.clear()
        ax.plot(xs, d0_)
        ax.plot(xs, d1_)
        ax.plot(xs, d2_)
        # Format plot
        # ax.set_ylim([0, 255])
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.20)
        # ax.set_title('Plot of random numbers from https://qrng.anu.edu.au')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Thickness (um)')

    # Set up plot to call animate() function every 1000 milliseconds
    ani = animation.FuncAnimation(fig, animate, fargs=(xs, d0_, d1_, d2_), interval=1)

    plt.show()

# data = r"\xfc\xf6\xfe\xfe\xfe\xfe\xfe\xfa"
#data = r"\xf3\x00\xb1\x8a\x56\xe0\xff\xff"
data0 = r"\xf3\x00\xb1\x8a\x56\xc0\x40\xf9"
data1 = r"\xf3\x00\xb1\x8a\x56\xe0\x40\xf9"
data2 = r"\xf3\x00\xb1\x8a\x56\x00\x41\xf9"
data3 = r"\xf3\x00\xb1\x8a\x56\x80\x4d\xfc"
data4 = r"\xf3\x00\xb1\x8a\x56\xc0\x9f\xfc"
data5 = r"\xf3\x00\xb1\x8a\x56\xe0\x8e\xf8"

data6 = r"\xf3\x00\xb1\x8a\x56\xc0\x3d\xad"

for data in [data6]:
    input_str = '0x' + "".join(list(reversed(data.split(r"\x"))))
    #print(input_str)

    c = BitArray(hex=input_str)
    #print(c.bin)


# TODO https://pythonforundergradengineers.com/live-plotting-with-matplotlib.html
# print(hex(int("1111101011111110111111101111111011111110111111101111011011111100", 2)))

with open('test', 'rb') as f:

    hexdata = binascii.hexlify(f.read())
    print(hexdata)

    # hexdata = hexdata[5:]


    cntr = 0
    for i in range(len(hexdata)//16 - 1):
        s_ = hexdata[i*16:(i+1)*16].decode()
        # print(s_)
        s = [s_[i:i + 2] for i in range(0, len(s_), 2)]
        input_str = '0x' + "".join(list(reversed(s)))
        # print(input_str)
        c = BitArray(hex=input_str)
        c_bin = str(c.bin)
        print(c_bin)
        d0, d1, d2 = c_bin[-45:-30], c_bin[-30:-15], c_bin[-15:]
        d0, d1, d2 = c_*int(d0, 2), c_*int(d1, 2), c_*int(d2, 2)
        print(d0, d1, d2)

        d0_.append(d0), d1_.append(d1), d2_.append(d2)


        cntr += 1
    print(cntr)

    plt.plot(d0_)
    plt.plot(d1_)
    plt.plot(d2_)
    plt.show()
