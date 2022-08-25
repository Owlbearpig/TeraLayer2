import numpy as np
import matplotlib.pyplot as plt
from simulation.tmm import get_phase, get_amplitude
from raw_phase import get_measured_phase, get_measured_amplitude
from consts import um_to_m, THz, array, GHz, pi
from visualizing.simplecolormap import map_plot

#sam_idx = 28
sam_idx = 78
#freqs = array([0.365, 0.503, 0.520, 1.087, 1.298, 1.380]) * THz
freqs = array([0.600, 0.642, 0.692, 0.772, 0.830, 0.856]) * THz
#freqs = np.arange(0.125, 1.650+0.001, 0.01/10) * THz
phase_measured = get_measured_phase(freqs, sam_idx)
amplitude_measured = get_measured_amplitude(freqs, sam_idx)
#freq_slice = (0.23 * THz <= freqs) * (freqs <= 1.80 * THz)

limited_slice = np.abs(phase_measured) <= pi
phase_measured = phase_measured[limited_slice]
freqs = freqs[limited_slice]

def phase_loss(p):
    phase_sim = get_phase(freqs, p)

    return np.sum((phase_sim - phase_measured) ** 2)


def total_loss(p):
    phase_sim = get_phase(freqs, p)
    amplitude_sim = get_amplitude(freqs, p)

    return np.sum(((phase_sim - phase_measured) ** 2) * ((amplitude_sim - amplitude_measured) ** 2))

plt.figure("Sim vs measurement")
plt.plot(freqs/GHz, phase_measured, label="Phase measured", color="black")

# p = np.array([60, 620, 75]) * um_to_m
p0 = np.array([45, 420, 75]) * um_to_m
p1 = np.array([51, 608, 56]) * um_to_m
p2 = np.array([26, 573, 91]) * um_to_m
p3 = np.array([40, 640, 75]) * um_to_m
p4 = np.array([46, 147, 915]) * um_to_m
p5 = np.array([41, 644, 157]) * um_to_m
p6 = np.array([583, 980, 111]) * um_to_m
p7 = np.array([41, 633, 924]) * um_to_m
p8 = np.array([31, 307, 518]) * um_to_m
solutions = [p3, p4, p5, p7, p8]
for p in solutions:
    phase_model = get_phase(freqs, p)

    print(p*10**6, phase_model, phase_loss(p))

    plt_label = f"Phase model {p * 10 ** 6}, phase_loss {round(float(phase_loss(p)), 2)}"
    plt.plot(freqs/GHz, phase_model, label=plt_label)

plt.xlabel("Frequency (GHz)")
plt.ylabel("Phase (rad)")
plt.legend()
plt.show()

if __name__ == '__main__':
    file_name = "total_loss_6freq_grid_vals_v1_0_0_0"

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
                print(f'j: {j}/{rez_x}')
                for k in range(rez_z):
                    p = array([grd_x[i], grd_y[j], grd_z[k]])
                    grid_vals[i, j, k] = phase_loss(p)

        np.save(file_name, grid_vals)

    map_plot(img_data=grid_vals)
