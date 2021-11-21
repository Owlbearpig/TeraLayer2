import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from multir_numba import multir_numba
from multir import multir
from consts import default_mask
from functions import format_data, avg_runtime

lam, R = format_data()

d = array([0.0000378283, 0.0006273254, 0.0000378208])

avg_runtime(multir, lam[default_mask], d)
# numba is like C implementation ?
avg_runtime(multir_numba, lam[default_mask], d)


plt.plot(lam/1e-3, R)
plt.plot(lam[default_mask]/1e-3, R[default_mask], 'o', color='red')
plt.plot(lam/1e-3, multir(d, lam))
plt.xlim((0, 2))
plt.ylim((0, 1.1))
plt.xlabel('THZ-Wavelenght (mm)')
plt.ylabel('$r^2$ (arb. units)')
plt.show()
