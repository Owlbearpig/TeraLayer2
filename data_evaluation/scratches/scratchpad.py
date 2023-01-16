from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numfi import numfi as numfi_
from functools import partial
from numpy import pi as pi64
from numpy import array

res = np.linspace(-100, 100, 10000)

res[res < -pi64] += 2*pi64
res[pi64 < res] -= 2*pi64
#res[res < 0] += 2*pi64
#res[res > pi64] -= 2*pi64

"""
x < -pi     : 2pi
-pi < x < 0 : x
0 < x < pi  : x
pi < x      : -2pi

x < 0 += 2pi
x > pi -= 2pi
"""



plt.plot(np.linspace(-100, 100, 10000), res)
plt.show()




