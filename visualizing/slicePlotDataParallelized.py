import numpy as np
from numpy import array
from functions import format_data, residuals, avg_runtime
from consts import um, custom_mask, default_mask, full_range_mask
from model.multir_numba import multir_numba
import multiprocessing
from joblib import Parallel, delayed
from pathlib import Path

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


def processInput(p):
    return sum(residuals(p, multir_numba, lam, R)).real

"""
def calc_grid():
    grid_vals = np.zeros([rez_x, rez_y, rez_z])
    for i in range(rez_x):
        for j in range(rez_y):
            for k in range(rez_z):
                p = array([grd_x[i], grd_y[j], grd_z[k]])
                grid_vals[i, j, k] = processInput(p)
    return grid_vals

#avg_runtime(calc_grid)
#np.save(f'{rez_x}_{rez_y}_{rez_z}_rez_xyz_cubed_grid-lb_ub_edges.npy', grid_vals)
"""

def make_inputs():
    inputs = np.zeros((rez_x*rez_y*rez_z, 3))
    input_idx = 0
    for i in range(rez_x):
        print(i)
        for j in range(rez_y):
            for k in range(rez_z):
                inputs[input_idx] = grd_x[i], grd_y[j], grd_z[k]
                input_idx += 1

    return inputs
#inputs = make_inputs()
#np.save('flat_input_grid_1000x1000x1000entries.npy', inputs)
#exit()

full_input_grid = np.load('flat_input_grid_1000x1000x1000entries.npy')

def calc_grid_parallel(batch):
    num_cores = 10 #multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in batch)

    return results


batch_cnt = 100
batch_len = len(full_input_grid) // batch_cnt  # ! has to be divisible int for this to work
for i in range(batch_cnt):
    print(f'{i}/100')
    res = calc_grid_parallel(full_input_grid[batch_len*i:batch_len*(i+1)])
    np.save(str(Path('grid_data') / 'fullgrid_fullrange' /f'batch_{i}.npy'), res)

