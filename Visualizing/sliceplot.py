import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from functions import format_data, residuals
from consts import default_mask, um
from multir_numba import multir_numba
from matplotlib.widgets import Slider, Button

"""
1. calculate sum(residuals) over 3D grid with some resolution(rez)
2. 2D plot slices for different z set with slider
"""

lam, R = format_data(mask=default_mask)

# should be resolution of axes d1, d2, d3
rez_x, rez_y, rez_z = 500, 500, 500

lb = array([0.000001, 0.00001, 0.000001])
ub = array([0.001, 0.001, 0.001])

# initial 'full' grid matching bounds
grd_x = np.linspace(lb[0], ub[0], rez_x)
grd_y = np.linspace(lb[1], ub[1], rez_y)
grd_z = np.linspace(lb[2], ub[2], rez_z)

"""
grid_vals = np.zeros([rez_x, rez_y, rez_z])
for i in range(rez_x):
    print(f'{i}/{rez_x}')
    for j in range(rez_y):
        for k in range(rez_z):
            p = array([grd_x[i], grd_y[j], grd_z[k]])
            grid_vals[i, j, k] = sum(residuals(p, multir_numba, lam, R))

np.save(f'{rez_x}_{rez_y}_{rez_z}_rez_xyz_cubed_grid-lb_ub_edges.npy', grid_vals)

"""
grid_vals = np.load('500_500_500_rez_xyz_cubed_grid-lb_ub_edges.npy')
grid_vals = np.log10(grid_vals)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Residual sum plot')
fig.subplots_adjust(left=0.2)
extent = [grd_x[0] * um, grd_x[-1] * um, grd_y[0] * um, grd_y[-1] * um]
img = ax.imshow(grid_vals[:, :, 0].transpose((1, 0)), vmin=np.min(grid_vals), vmax=np.max(grid_vals), origin='lower',
                extent=extent)  # , cmap=plt.get_cmap('autumn')
ax.set_xlabel('$d_1$')
ax.set_ylabel('$d_2$')

g_min_idx = np.argmin(grid_vals)
min_x, min_y, min_z = np.unravel_index(g_min_idx, grid_vals.shape)
print(grd_x[min_x] * um, grd_y[min_y] * um, grd_z[min_z] * um)
print(np.min(grid_vals))

fig.colorbar(img)

axmax = fig.add_axes([0.05, 0.1, 0.02, 0.8])
amp_slider = Slider(
    ax=axmax,
    label='$d_3$',
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

