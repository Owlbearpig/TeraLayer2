import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from multir import multir
from multir_numba import multir_numba
from consts import default_mask, um
from functions import format_data, residuals, avg_runtime
from scipy.optimize import least_squares

lam, R = format_data(mask=default_mask)

d_goal = array([0.0000378283, 0.0006273254, 0.0000378208])

d0 = array([0.000045, 0.00060, 0.000045])
lb = array([0.000001, 0.00001, 0.000001])
hb = array([0.001, 0.001, 0.001])

res = least_squares(residuals, d0, bounds=(lb, hb), args=(multir_numba, lam, R))
print(res.x * um)

# avg_runtime(least_squares, residuals, d0, bounds=(lb, hb), args=(multir_numba, lam, R))
