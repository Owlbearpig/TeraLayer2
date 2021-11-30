import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from functions import format_data, residuals
from consts import um, custom_mask, default_mask, full_range_mask
from model.multir_numba import multir_numba
from matplotlib.widgets import Slider

"""
1. calculate sum(residuals) over 3D grid with some resolution(rez)
"""

lam, R = format_data(mask=full_range_mask)

# should be resolution of axes d1, d2, d3
rez_x, rez_y, rez_z = 1000, 1000, 1000

#lb = array([0.000001, 0.000575, 0.000001])
#ub = array([0.000100, 0.000675, 0.000100])
lb = array([0.000001, 0.000001, 0.000001])
ub = array([0.001, 0.001, 0.001])

# initial 'full' grid matching bounds
grd_x = np.linspace(lb[0], ub[0], rez_x)
grd_y = np.linspace(lb[1], ub[1], rez_y)
grd_z = np.linspace(lb[2], ub[2], rez_z)


grid_vals = np.zeros([rez_x, rez_y, rez_z])
for i in range(rez_x):
    print(f'{i}/{rez_x}')
    for j in range(rez_y):
        for k in range(rez_z):
            p = array([grd_x[i], grd_y[j], grd_z[k]])
            grid_vals[i, j, k] = sum(residuals(p, multir_numba, lam, R))

np.save(f'{rez_x}_{rez_y}_{rez_z}_rez_xyz_cubed_grid-lb_ub_edges.npy', grid_vals)
