from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numfi import numfi as numfi_
from functools import partial
from numpy import pi as pi64
from numpy import array

pd, p = 4, 11
numfi = partial(numfi_, s=1, w=pd+p, f=p, fixed=True, rounding='floor')

pi = numfi(pi64)
pi2 = numfi_(2*pi64, s=1, w=6+p, f=p, fixed=True, rounding='floor')

def c_mod(s):
    """
    should do (s % 2pi) and if res is > pi subtract 2pi
    max in = 2*0.02986579156281*1000 + 0.055749477583909995 * 1000 / (2*pi*2**5) = 0.574
    max out = \pm pi
    """

    # s_scaled = s / (2 * pi64 * 2 ** 5)

    s_fp = numfi_(array(s), s=1, w=12 + p, f=p, fixed=True, rounding='floor')

    s_int = (s_fp << 5).astype(int)

    s_interm = (s_fp << 5) - s_int

    res = pi2 * numfi(s_interm)

    res[res < 0] += pi2
    res[res > pi] -= pi2

    return res

n = 1000 / (2*pi64*2**5)
#n = numfi(n)
g = 0.055749477583909995
f = 0.02986579156281
f0, f1, f2 = n*f, n*g, n*f
s0 = f0 + f1 + f2
print(s0)
print(c_mod(s0))
print(n)
b0 = numfi(1.23456789)




