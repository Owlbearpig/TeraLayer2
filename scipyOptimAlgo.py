import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from multir import multir
from multir_numba import multir_numba
from multir_1D import multir_1D
from consts import default_mask, default_mask_hr, um, um_to_m, wide_mask, custom_mask
from functions import format_data, residuals, avg_runtime, calc_scipy_loss, calc_loss
from Visualizing.plotting import plot_result
from scipy.optimize import least_squares

chosen_mask = custom_mask

lam, R = format_data(mask=chosen_mask, sample_file_idx=0)

d_goal = array([0.0000378283, 0.0006273254, 0.0000378208])

d0=array([0.000045, 0.00060, 0.000045])
print(d0*um)
#d0 = array([0.00050, 0.00050, 0.00050])
lb = array([0.000001, 0.00001, 0.000001])
hb = array([0.001, 0.001, 0.001])

"""
d0 = array([0.0006273254])
lb = array([0.00001])
hb = array([0.001])
"""

res = least_squares(residuals, d0, bounds=(lb, hb), args=(multir_numba, lam, R))
res_x = res.x * um
print(res)
print(res_x)
print(calc_loss(array([37, 626, 37])*um_to_m))

plot_result(res.x, mask=chosen_mask)
#plot(array([0.000045, 0.00060, 0.000045]))
# print(calc_loss(res.x))
# print(calc_loss(p_brutef))

# avg_runtime(least_squares, residuals, d0, bounds=(lb, hb), args=(multir_numba, lam, R))
