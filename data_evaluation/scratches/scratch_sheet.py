from numpy import array
import numpy as np
from scipy.constants import c
from consts import um_to_m, GHz

freq = 1400000000000.0

M = array([[0.04097605 + 1.53389583j, 0. - 1.16383652j], [0. + 1.16383652j, 0.04097605 - 1.53389583j]])

lam = (c/freq)
k = 2*np.pi/lam

r_wiki = ((M[1, 0] + k * k * M[0, 1]) + 1j * (k * M[1, 1] - k * M[0, 0])) / (
            (-M[1, 0] + k * k * M[0, 1]) + 1j * (k * M[1, 1] + k * M[0, 0]))

r_ecc = M[0, 1] / M[1, 1]
r_nina = M[1, 0] / M[0, 0]

print(r_nina)
print(r_wiki)
print(r_ecc)

d = 600 * um_to_m
n = 2.7

print((c/(d*n*2))/GHz)


