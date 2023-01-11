from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numfi import numfi as numfi_
from functools import partial
from numpy import pi as pi64
from numpy import array

pd, p = 4, 21
numfi = partial(numfi_, s=1, w=pd+p, f=p, fixed=True, rounding='floor')

b0 = numfi(pi64)
pi = numfi(pi64)
pi2 = numfi_(2*pi64, s=1, w=6+p, f=p, fixed=True, rounding='floor')

#s = numfi(0.1328491)

def sine(x):
    # print(x)
    # x -= (x > pi) * (2 * pi)
    # x = 1.337

    B = numfi(4 / pi64)
    C = numfi(-4 / (pi64 * pi64))

    y = x * (B + C * np.abs(x))

    # print('y', y)
    P = numfi(0.225)
    res = P * y * (np.abs(y) - 1) + y

    return res

def c_mod_correct(s):
    """
    m = (s / (2*pi64)).astype(int)

    res = s - m * 2*pi64
    res[res < 0] += 2*pi64

    res[res > pi64] -= 2 * pi64
    """
    #s = s / (2 * pi64 * 2 ** 6)
    n = 2 ** 6
    s /= 2 * pi64 * n

    res = 2 * pi64 * (s * n - (s * n).astype(int))

    res[res < 0] += 2 * pi64

    res[res > pi64] -= 2 * pi64
    """
    res = s -  2 * pi64 * (s / (2 * pi64)).astype(int)
    res[res < 0] += 2 * pi64

    res[res > pi64] -= 2 * pi64
    """
    return res

def c_mod(s):
    s_scaled = s / (2*pi64*2**6)

    s_fp = numfi_(array(s_scaled), s=1, w=12+p, f=p, fixed=True, rounding='floor')

    s_int = (s_fp << 6).astype(int)

    s_interm = (s_fp << 6) - s_int

    res = pi2 * numfi(s_interm)

    res[res < 0] += pi2
    res[res > pi] -= pi2

    return res


s = np.linspace(-14, 14, 1000)

y1 = c_mod_correct(s)
s = np.linspace(-14, 14, 1000)
y = (s % (2*pi64))
y[y > pi64] -= 2*pi

plt.figure()
plt.plot(s, y)
plt.plot(s, y1)
s = np.linspace(-14, 14, 1000)
plt.plot(np.linspace(-14, 14, 1000), c_mod(s))

plt.figure()
#plt.scatter(y1, sine(y1))
s = np.linspace(-14, 14, 1000)
s_mod = c_mod(s)
sinee = sine(s_mod)

plt.plot(s, sinee)
plt.plot(s, np.sin(s))
plt.show()

