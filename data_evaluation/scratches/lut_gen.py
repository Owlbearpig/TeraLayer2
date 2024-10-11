from numfi import numfi as numfi_
from functools import partial
import numpy as np
from numpy import sin, pi
import matplotlib.pyplot as plt

pd = 4
p = 11
lut_table = f"sin_lut_{p}"

numfi = partial(numfi_, s=1, w=pd + p, f=p, rounding="floor", fixed=True)

with open(lut_table, "w") as file:
    for i, x in enumerate(np.linspace(-pi, pi, 2**(pd+p))):
        sin_x = numfi_(sin(x), s=1, w=2+p, f=p, rounding="floor")
        s = sin_x.bin[0] + f" // {x} {numfi(x).bin[0]} {i:05}: sin(x) = {sin(x)} \n"  # Needs whitespace before comment?
        # s = sin_x.bin[0] + "\n"
        file.write(s)

numfi = partial(numfi_, s=1, w=pd+p, f=p, rounding="floor", fixed=True)
pi2_inv = numfi_(1/(2 * pi), s=1, w=pd+p, f=p, rounding="floor", fixed=True)
half = numfi(0.5)
print(pi2_inv.bin, half.bin)


def sin_lut(x):
    print(x, numfi(x).bin)
    addr = (numfi(x) * pi2_inv + half) << 4
    print(addr, addr.bin)
    addr = int(addr.bin[0], 2)
    print(addr)

    with open(lut_table, "r") as file:
        lines = file.readlines()

    print(lines[addr])

    if lines[addr][0] == "1":
        return int(lines[addr][0:2 + p], 2) * 2 ** (-p) - 2**2

    return int(lines[addr][0:2 + p], 2) * 2 ** (-p)


print(sin_lut(-1.791992))
print(np.sin(pi*0.1))
exit()
x_arr = np.linspace(-pi, pi, 2**11)
lut_res = []
for x in x_arr:
    lut_res.append(sin_lut(x))

print(np.sum(np.array(lut_res) - np.sin(x_arr)))

plt.plot(x_arr, lut_res, label="lut")
plt.plot(x_arr, np.sin(x_arr), label="np.sin")
plt.ylabel("sin(x)")
plt.xlabel("x")
plt.legend()
plt.show()
