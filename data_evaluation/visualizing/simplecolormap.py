import matplotlib.pyplot as plt
from consts import *
from matplotlib.widgets import Slider


def map_plot(error_func=None, img_data=None, settings=None, representation="", title=""):

    if settings is None:
        # should be resolution of axes d1, d2, d3
        rez_x, rez_y, rez_z = 200, 200, 200
        # rez_x, rez_y, rez_z = 1000, 1000, 1000

        # lb = array([0.000001, 0.000400, 0.000001]) # realistic bounds
        # ub = array([0.000100, 0.000700, 0.000100])
        lb = array([0.000001, 0.000001, 0.000001])
        ub = array([0.001, 0.001, 0.001])
        unit_lbl = "$(\mu m)$"
        unit = um
    else:
        rez_x, rez_y, rez_z = settings["rez"]
        lb, ub = settings["lb"], settings["ub"]
        unit_lbl = settings["unit_lbl"]
        unit = settings["unit"]

    # initial 'full' grid matching bounds
    grd_x = np.linspace(lb[0], ub[0], rez_x)
    grd_y = np.linspace(lb[1], ub[1], rez_y)
    grd_z = np.linspace(lb[2], ub[2], rez_z)

    grid_vals = np.zeros([rez_x, rez_y, rez_z])
    if (img_data is not None) or (error_func is None):
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
    if representation == "log":
        grid_vals = np.log10(grid_vals)
        cbar_label = "log10(loss)"
    elif representation == "recip":
        grid_vals = -1/(grid_vals)
        cbar_label = "-1/(loss)"
    elif representation == "both":
        grid_vals = -np.log10(1 / (grid_vals))
        cbar_label = "-log10(1/loss)"
    else:
        cbar_label = "loss value"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if title is None:
        ax.set_title('Residual sum plot')
    else:
        ax.set_title(title)
    fig.subplots_adjust(left=0.2)
    extent = [grd_x[0] * unit, grd_x[-1] * unit, grd_y[0] * unit, grd_y[-1] * unit]
    img = ax.imshow(grid_vals[:, :, 0].transpose((1, 0)), vmin=np.min(grid_vals), vmax=np.max(grid_vals), origin='lower',
                    cmap=plt.get_cmap('jet'),
                    extent=extent)

    ax.set_xlabel(f'$p_1$ {unit_lbl}')
    ax.set_ylabel(f'$p_2$ {unit_lbl}')

    g_min_idx = np.argmin(grid_vals_og)
    min_x, min_y, min_z = np.unravel_index(g_min_idx, grid_vals_og.shape)
    print(grd_x[min_x] * unit, grd_y[min_y] * unit, grd_z[min_z] * unit)
    print(np.min(grid_vals_og))

    cbar = fig.colorbar(img)

    cbar.set_label(cbar_label, rotation=270, labelpad=15)
    axmax = fig.add_axes([0.05, 0.1, 0.02, 0.8])
    amp_slider = Slider(
        ax=axmax,
        label=f'$p_3$ {unit_lbl}',
        valstep=grd_z * unit,
        valmin=grd_z[0] * unit,
        valmax=grd_z[-1] * unit,
        valinit=grd_z[0] * unit,
        orientation='vertical'
    )


    def update(val):
        idx, = np.where(grd_z * unit == val)
        img.set_data(grid_vals[:, :, idx].transpose((1, 0, 2)))
        fig.canvas.draw()


    amp_slider.on_changed(update)
    plt.show()

    return grid_vals_og
