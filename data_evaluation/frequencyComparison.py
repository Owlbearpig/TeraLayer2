import time

from numpy import array, sum
import numpy as np
from model.multir_numba import multir_numba
from consts import um, um_to_m, custom_mask, default_mask, full_range_mask
from functions import format_data
from visualizing.plotting import plot_result, plot_R
from optimization.nelderMeadSource import _minimize_neldermead
import matplotlib.pyplot as plt

res = []
for mask_shift in range(100):
    lam, R = format_data(mask=full_range_mask+mask_shift, sample_file_idx=10)

    def error(p):
        return sum((multir_numba(lam, p).real - R.real) ** 2)


    d0 = array([30, 600, 30]) * um_to_m
    lb = d0 - array([20, 50, 20]) * um_to_m
    hb = d0 + array([20, 50, 20]) * um_to_m

    fval, x, iterations, fcalls = _minimize_neldermead(error, d0, bounds=(lb, hb), adaptive=False)
    res.append(x)

res = np.array(res)

print(np.std(res[:, 0])*um, np.std(res[:, 1])*um, np.std(res[:, 2])*um)

plt.plot(res[:, 0]*um)
plt.plot(res[:, 1]*um)
plt.plot(res[:, 2]*um)
plt.show()

