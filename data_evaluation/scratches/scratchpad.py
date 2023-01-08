from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numfi import numfi as numfi_
from functools import partial
from numpy import pi as pi64
from numpy import array

pd, p = 4, 23
numfi = partial(numfi_, s=1, w=pd+p, f=p, fixed=True, rounding='floor')

b0 = numfi(pi64)
pi = numfi(pi64)
pi2 = numfi_(2*pi64, s=1, w=6+p, f=p, fixed=True, rounding='floor')

s = numfi(0.1328491)

def c_mod(s):
    print(s.bin)
    s = numfi_(array(s), s=1, w=12+p, f=p, fixed=True, rounding='floor')
    print(s.bin)
    s_int = (s << 6).astype(int)
    print(s_int)
    s_interm = (s << 6) - s_int
    print(s_interm, "\n")
    res = pi2 * s_interm

    #res[res > pi] -= pi2

    return res


print(c_mod(s))

