import numpy as np
from numpy import array, sum
import matplotlib.pyplot as plt
from functions import format_data, residuals
from consts import *
from model.multir_numba import multir_numba
from model.explicitEvalOptimized import explicit_reflectance
from matplotlib.widgets import Slider
from model.explicitEvalOptimizedClean import ExplicitEval


def map_plot(error_func, img_data=None):

    # should be resolution of axes d1, d2, d3
    rez_x, rez_y, rez_z = 200, 200, 200
    # rez_x, rez_y, rez_z = 1000, 1000, 1000

    # lb = array([0.000001, 0.000400, 0.000001]) # realistic bounds
    # ub = array([0.000100, 0.000700, 0.000100])
    lb = array([0.000001, 0.000001, 0.000001])
    ub = array([0.001, 0.001, 0.001])

    # initial 'full' grid matching bounds
    grd_x = np.linspace(lb[0], ub[0], rez_x)
    grd_y = np.linspace(lb[1], ub[1], rez_y)
    grd_z = np.linspace(lb[2], ub[2], rez_z)

    grid_vals = np.zeros([rez_x, rez_y, rez_z])
    if img_data is not None:
        grid_vals = img_data.copy()
    else:
        for i in range(rez_x):
            if (i % 5) == 0:
                print(f'{i}/{rez_x}')
            for j in range(rez_y):
                for k in range(rez_z):
                    p = array([grd_x[i], grd_y[j], grd_z[k]])
                    grid_vals[i, j, k] = error_func(p)
                print(i, j)

    grid_vals_og = grid_vals.copy()
    grid_vals = np.log10(grid_vals)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Residual sum plot')
    fig.subplots_adjust(left=0.2)
    extent = [grd_x[0] * um, grd_x[-1] * um, grd_y[0] * um, grd_y[-1] * um]
    img = ax.imshow(grid_vals[:, :, 0].transpose((1, 0)), vmin=np.min(grid_vals), vmax=np.max(grid_vals), origin='lower',
                    cmap=plt.get_cmap('jet'),
                    extent=extent)

    ax.set_xlabel('$d_1$ $(\mu m)$')
    ax.set_ylabel('$d_2$ $(\mu m)$')

    g_min_idx = np.argmin(grid_vals)
    min_x, min_y, min_z = np.unravel_index(g_min_idx, grid_vals.shape)
    print(grd_x[min_x] * um, grd_y[min_y] * um, grd_z[min_z] * um)
    print(np.min(grid_vals))

    cbar = fig.colorbar(img)
    cbar.set_label('log10(loss)', rotation=270, labelpad=10)
    axmax = fig.add_axes([0.05, 0.1, 0.02, 0.8])
    amp_slider = Slider(
        ax=axmax,
        label='$d_3$ $(\mu m)$',
        valstep=grd_z * um,
        valmin=grd_z[0] * um,
        valmax=grd_z[-1] * um,
        valinit=grd_z[0] * um,
        orientation='vertical'
    )


    def update(val):
        idx, = np.where(grd_z * um == val)
        img.set_data(grid_vals[:, :, idx].transpose((1, 0, 2)))
        fig.canvas.draw()


    amp_slider.on_changed(update)
    plt.show()

    return grid_vals_og
