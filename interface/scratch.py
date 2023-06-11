from bitstring import BitArray
import binascii
from numpy import pi
import matplotlib.pyplot as plt

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

# print(hex(int("1111101011111110111111101111111011111110111111101111011011111100", 2)))
c_ = 2 ** 6 * 2 * pi * 2 ** (-11)
with open('test', 'rb') as f:

    hexdata = binascii.hexlify(f.read())
    print(hexdata)

    # hexdata = hexdata[5:]

    d0_, d1_, d2_ = [], [], []
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
