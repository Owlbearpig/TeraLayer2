import numpy as np
import matplotlib.pyplot as plt
from model.tmm import get_phase, get_amplitude
from model.itsjustaphase.raw_phase import get_measured_phase, get_measured_amplitude
from consts import um_to_m, THz, array, GHz, pi, ones
from visualizing.simplecolormap import map_plot
from helpers import is_iterable

# sam_idx = 28
sam_idx = 78
# freqs = array([0.365, 0.503, 0.520, 1.087, 1.298, 1.380]) * THz
#freqs = array([0.600, 0.642, 0.692, 0.772, 0.830, 0.856]) * THz
#freqs = array([0.610, 0.636, 0.694, 0.770, 0.830, 0.870]) * THz
#freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950]) * THz
#freqs = array([0.374, 0.416, 0.470, 0.596, 0.644, 0.718]) * THz # Doesnt work
freqs = array([0.348, 0.407, 0.485, 0.514, 0.567, 0.600]) * THz
#freqs = np.arange(0.050, 1.200 + 0.001, 0.001) * THz
phase_measured = get_measured_phase(freqs, sam_idx)
amplitude_measured = get_measured_amplitude(freqs, sam_idx)
# freq_slice = (0.23 * THz <= freqs) * (freqs <= 1.80 * THz)

limited_slice = np.abs(phase_measured) <= pi
phase_measured = phase_measured[limited_slice]
amplitude_measured = amplitude_measured[limited_slice]
freqs = freqs[limited_slice]


def get_n(freqs):
    m = len(freqs)
    n_min, n_max = 2.67, 2.80
    a = (n_max - n_min) / m

    n1 = np.arange(0, m)*a + n_min

    n = np.array([ones(m), 1.50*ones(m), n1, 1.50*ones(m), ones(m)], dtype=np.complex128).transpose()

    return n


n = get_n(freqs)


def phase_loss(p):
    phase_sim = get_phase(freqs, p, n)

    return np.sum((phase_sim - phase_measured) ** 2)


def amplitude_loss(p):
    amplitude_sim = get_amplitude(freqs, p, n)

    return np.sum((amplitude_sim - amplitude_measured) ** 2)


def total_loss(p):
    p_loss = phase_loss(p)
    amp_loss = amplitude_loss(p)

    return p_loss * amp_loss

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(freqs / GHz, phase_measured, label="Phase measured", color="black")
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("Phase (rad)")

ax2.plot(freqs / GHz, amplitude_measured, label="Amplitude measured", color="black")
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Reflectance")
fig.suptitle("Model vs measurement")

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
p9 = np.array([31, 639, 71]) * um_to_m
p10 = np.array([734, 1000, 458]) * um_to_m
p11 = np.array([513, 739, 684]) * um_to_m
p12 = np.array([31, 563, 71]) * um_to_m
p13 = np.array([21, 895, 573]) * um_to_m

solutions = array([p3, p5])
for p in solutions:
    phase_model = get_phase(freqs, p, n)
    amplitude_model = get_amplitude(freqs, p, n)

    print(p * 10 ** 6, total_loss(p))

    plt_label_phase = f"Phase model {p * 10 ** 6}, p_loss {round(float(phase_loss(p)), 6)}"
    ax1.plot(freqs / GHz, phase_model, label=plt_label_phase)

    plt_label_amplitude = f"Amp. model {p * 10 ** 6}, a_loss {round(float(amplitude_loss(p)), 6)}"
    ax2.plot(freqs / GHz, amplitude_model, label=plt_label_amplitude)

ax1.legend()
ax2.legend()

best_sol = solutions[np.argmin(array([total_loss(p) for p in solutions]))]
best_min_p, best_min_a = phase_loss(best_sol), amplitude_loss(best_sol)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Difference between model and measurement")
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("Phase (rad)")

ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Reflectance")

for p in solutions:
    phase_model, amplitude_model = get_phase(freqs, p, n), get_amplitude(freqs, p, n)
    diff_phase, diff_amplitude = (phase_measured - phase_model)**2, (amplitude_measured - amplitude_model)**2

    plt_label_phase = f"Phase difference {p * 10 ** 6}, p_loss {round(float(phase_loss(p)), 6)}"
    ax1.plot(freqs / GHz, diff_phase, label=plt_label_phase)

    plt_label_amplitude = f"Amp. difference {p * 10 ** 6}, a_loss {round(float(amplitude_loss(p)), 6)}"
    ax2.plot(freqs / GHz, diff_amplitude, label=plt_label_amplitude)

ax1.legend()
ax2.legend()
plt.show()

print(f"Best solution: {best_sol * 10 ** 6}")
print(f"Total_loss = loss_p*loss_a: {best_min_p}*{best_min_p}={best_min_a * best_min_p}")



if __name__ == '__main__':
    file_name = "total_loss_6freq_grid_vals_v1_0_1_0"

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
                if (j % 10) == 0:
                    print(f'j: {j}/{rez_x}')
                for k in range(rez_z):
                    p = array([grd_x[i], grd_y[j], grd_z[k]])
                    grid_vals[i, j, k] = phase_loss(p)

        np.save(file_name, grid_vals)

    map_plot(img_data=grid_vals)
