import time
import numpy as np
from numpy import array, sum
from functions import format_data, residuals, avg_runtime
from consts import um, custom_mask, default_mask, full_range_mask
from model.multir_numba import multir_numba
from model.explicitEvalOptimized import explicit_reflectance
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


def prepare_inputs():
    inputs = np.zeros((rez_x*rez_y*rez_z, 3))
    input_idx = 0
    for i in range(rez_x):
        print(i)
        for j in range(rez_y):
            for k in range(rez_z):
                inputs[input_idx] = grd_x[i], grd_y[j], grd_z[k]
                input_idx += 1

    return inputs

#np.save('flat_input_grid_10x1000x1000entries.npy', full_input_grid)


def processBatch(p_arr):
    return array([sum((explicit_reflectance(p).real-R.real)**2) for p in p_arr])


full_input_grid = np.load('flat_input_grid_1000x1000x1000entries.npy', mmap_mode='r')


def calc_grid_parallel(batch):
    num_cores = 10#  multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores, verbose=1)(delayed(processBatch)(k) for k in np.array_split(batch, num_cores))

    return array(results)


batch_cnt = 100  # split grid into 100 parts, process each part distributed to all(or num_cores) cpu cores.
all_batches = np.array_split(full_input_grid, batch_cnt)
for i, batch in enumerate(all_batches):
    print(f'Processing batch: {i}/{batch_cnt}')
    res = calc_grid_parallel(batch)
    np.save(str(Path('grid_data') / 'fullgrid_fullrange' / f'batch_{i}.npy'), res)
    print(f'Done {i}/{batch_cnt}.')
