import matplotlib.pyplot as plt
from numpy import array
from multir import multir
from consts import default_mask
from functions import format_data

"""
Should reproduce matlab/octave multir. Test if multir is implemented correctly.
"""

lam, R = format_data()

d = array([0.0000378283, 0.0006273254, 0.0000378208])

plt.plot(lam/1e-3, R)
plt.plot(lam[default_mask]/1e-3, R[default_mask], 'o', color='red')
plt.plot(lam/1e-3, multir(lam, d))
plt.xlim((0, 2))
plt.ylim((0, 1.1))
plt.xlabel('THZ-Wavelenght (mm)')
plt.ylabel('$r^2$ (arb. units)')
plt.show()
