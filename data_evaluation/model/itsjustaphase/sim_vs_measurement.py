import numpy as np
import matplotlib.pyplot as plt
from simulation.tmm import get_phase
from consts import um_to_m, THz, array
from visualizing.simplecolormap import map_plot

#freqs, phase_measured = np.load("freqs_measured_mean.npy"), np.load("phase_measured_mean.npy")
freqs, phase_measured = np.load("freqs_measured.npy"), np.load("phase_measured.npy")
freq_slice = (0.23 * THz <= freqs) * (freqs <= 1.80 * THz)
freqs, phase_measured = freqs[freq_slice][::10], phase_measured[freq_slice][::10]


def phase_loss(p):
    phase_sim = get_phase(freqs, p)

    loss = np.sum((phase_sim - phase_measured) ** 2)

    return loss


plt.figure("Phase unwrapped sim vs measurement")
plt.plot(freqs, phase_measured, label="Phase measured", color="black")

# p = np.array([60, 620, 75]) * um_to_m
p0 = np.array([45, 420, 75]) * um_to_m
p1 = np.array([51, 608, 56]) * um_to_m
p2 = np.array([26, 573, 91]) * um_to_m
p3 = np.array([40, 640, 75]) * um_to_m
solutions = [p0, p1, p2, p3]
for p in solutions:
    phase_sim = get_phase(freqs, p)

    print(p, phase_loss(p))

    plt.plot(freqs, phase_sim, label=f"Phase sim {p * 10 ** 6}, phase_loss {round(float(phase_loss(p)), 2)}")
plt.legend()
plt.show()
exit()

if __name__ == '__main__':
    file_name = "phase_loss_mean_grid_vals_1_0"

    try:
        grid_vals = np.load(file_name + ".npy")
    except FileNotFoundError:
        rez_x, rez_y, rez_z = 200, 200, 200
        lb = array([0.000001, 0.000001, 0.000001])
        ub = array([0.001, 0.001, 0.001])

        # initial 'full' grid matching bounds
        grd_x = np.linspace(lb[0], ub[0], rez_x)
        grd_y = np.linspace(lb[1], ub[1], rez_y)
        grd_z = np.linspace(lb[2], ub[2], rez_z)

        grid_vals = np.zeros([rez_x, rez_y, rez_z])
        for i in range(rez_x):
            print(f'i: {i}/{rez_x}')
            for j in range(rez_y):
                print(f'j: {j}/{rez_y}')
                for k in range(rez_z):
                    p = array([grd_x[i], grd_y[j], grd_z[k]])
                    grid_vals[i, j, k] = phase_loss(p)

        np.save(file_name, grid_vals)

    map_plot(img_data=grid_vals)
