from consts import *


def _minimize_bruteforce(fun, d0, bounds):
    rez_x, rez_y, rez_z = 100, 100, 100

    lb, ub = bounds

    grd_x = np.linspace(lb[0], ub[0], rez_x)
    grd_y = np.linspace(lb[1], ub[1], rez_y)
    grd_z = np.linspace(lb[2], ub[2], rez_z)

    grid_vals = np.zeros([rez_x, rez_y, rez_z])
    for i in range(rez_x):
        if (i % 10) == 0:
            print(f'{i}/{rez_x}')
        for j in range(rez_y):
            for k in range(rez_z):
                p = array([grd_x[i], grd_y[j], grd_z[k]])
                grid_vals[i, j, k] = fun(p)

    g_min_idx = np.argmin(grid_vals)
    min_x, min_y, min_z = np.unravel_index(g_min_idx, grid_vals.shape)

    return np.min(grid_vals), array([grd_x[min_x], grd_y[min_y], grd_z[min_z]]), 1, rez_x*rez_y*rez_z
