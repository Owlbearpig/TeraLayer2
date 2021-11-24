import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from multir import multir
#from multir_numba import multir_numba
from multir_1D import multir_1D
from consts import default_mask, um, um_to_m
from functions import format_data, residuals, avg_runtime, calc_scipy_loss, calc_loss, plot
from scipy.optimize import least_squares

lam, R = format_data(mask=default_mask)

d_goal = array([0.0000378283, 0.0006273254, 0.0000378208])

# d0 = array([0.000045, 0.00060, 0.000045])
# lb = array([0.000001, 0.00001, 0.000001])
# hb = array([0.001, 0.001, 0.001])

d0 = array([0.0006273254])
lb = array([0.00001])
hb = array([0.001])

res = least_squares(residuals, d0, bounds=(lb, hb), args=(multir_1D, lam, R))
print(res)
print(res.x * um)

#plot(res.x, fun=multir_1D)
plot(array([0.000045, 0.00060, 0.000045]))
# print(calc_loss(res.x))
# print(calc_loss(p_brutef))

# avg_runtime(least_squares, residuals, d0, bounds=(lb, hb), args=(multir_numba, lam, R))
