import numpy as np
import matplotlib.pyplot as plt
from model.tmm import get_phase, get_amplitude
from measurement_data import get_measured_phase, get_measured_amplitude
from consts import um_to_m, THz, array, GHz, pi, ones, c0
from visualizing.simplecolormap import map_plot
from refractive_index import get_n
from random import randint
from pathlib import Path
from scipy.optimize import curve_fit

# sam_idx = 59
sam_idx = 28
# sam_idx = randint(50, 100)

# freqs = array([0.365, 0.503, 0.520, 1.087, 1.298, 1.380]) * THz
# freqs = array([0.600, 0.642, 0.692, 0.772, 0.830, 0.856]) * THz
# freqs = array([0.610, 0.636, 0.694, 0.770, 0.830, 0.870]) * THz
# freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950]) * THz
# freqs = array([0.374, 0.416, 0.470, 0.596, 0.644, 0.718]) * THz # Doesnt work
# freqs = array([0.348, 0.407, 0.485, 0.514, 0.567, 0.600]) * THz # Does work if only considering phase loss...
# freqs = array([0.348, 0.507, 0.645, 0.764, 0.867, 0.970]) * THz # Does work for total loss
# freqs = array([0.191, 0.267, 0.345, 0.425, 0.500, 0.666]) * THz # does not work at all
# freqs = array([0.309, 0.386, 0.461, 0.551, 0.700, 0.882]) * THz
# freqs = array([0.346, 0.471, 0.562, 0.760, 0.964, 1.045]) * THz
# freqs = array([0.250, 0.420, 0.521, 0.610, 0.721, 0.780]) * THz
# freqs = array([0.560, 0.711, 1.120, 1.160, 1.240, 1.320]) * THz
# freqs = array([0.440, 0.520, 0.600, 0.680, 0.780, 0.860]) * THz
# freqs = array([7.66e+11, 7.96e+11, 8.26e+11, 8.56e+11, 8.86e+11, 9.26e+11]) + 10 * GHz
freqs = array([0.460, 0.490, 0.600, 0.640, 0.780, 0.840]) * THz
freqs = array([0.040, 0.070, 0.600, 0.640, 0.940, 0.960]) * THz
freqs = array([0.050, 0.060, 0.130, 0.540, 0.680, 0.720]) * THz
freqs = array([0.050, 0.070, 0.150, 0.600, 0.680, 0.720]) * THz
# freqs = array(np.random.randint(250, 1300, 6), dtype=np.float64)
# freqs *= GHz
# freqs.sort()
# print(freqs)
# freqs = np.arange(0.400, 1.400 + 0.001, 0.001) * THz
# freqs = np.arange(0.400, 0.600 + 0.001, 0.001) * THz
all_freqs = np.arange(0.001, 1.400 + 0.001, 0.001) * THz
phase_measured = get_measured_phase(freqs, sam_idx)
amplitude_measured = get_measured_amplitude(freqs, sam_idx)
# freq_slice = (0.23 * THz <= freqs) * (freqs <= 1.80 * THz)


limited_slice = np.abs(phase_measured) <= pi
phase_measured = phase_measured[limited_slice]
amplitude_measured = amplitude_measured[limited_slice]
freqs = freqs[limited_slice]

print(freqs / THz)
print(len(phase_measured))
if len(freqs) <= 6:
    # assert len(phase_measured) == 6, f"Correct freqs: {freqs / THz}"
    print(f"Correct freqs: {freqs / THz}")

n = get_n(freqs, 2.70, 2.85)

n = get_n(freqs, 2.70, 2.70)

phase_measured = get_phase(freqs, np.array([42.5, 641.3, 74.4]) * um_to_m, n)
amplitude_measured = get_amplitude(freqs, np.array([42.5, 641.3, 74.4]) * um_to_m, n)
phase_measured = get_phase(freqs, np.array([142.5, 541.3, 174.4]) * um_to_m, n)
amplitude_measured = get_amplitude(freqs, np.array([142.5, 541.3, 174.4]) * um_to_m, n)


def phase_loss(p):
    phase_sim = get_phase(freqs, p, n)

    return np.sum((1 / len(freqs)) * ((phase_sim - phase_measured) / (2 * pi)) ** 2)


def amplitude_loss(p):
    amplitude_sim = get_amplitude(freqs, p, n)

    return np.sum((1 / len(freqs)) * (amplitude_sim - amplitude_measured) ** 2)


def sine(x, a, omega):
    return np.abs(a * np.sin(x * omega))


def total_loss(p):
    p_loss = phase_loss(p)
    amp_loss = amplitude_loss(p)

    """
    # return ((np.sum(p) - np.sum(p_opt))*(1/3e-3))**2 + amp_loss*p_loss
    #return ((np.sum(p) - np.sum(p_opt))*(1/3e-3))**2 + amp_loss
    #return amp_loss
    #return p_loss

    selected_freqs = array([50, 60], dtype=float) * GHz
    selected_mod_points = get_amplitude(selected_freqs, p_opt, n)

    p0 = array([0.554, 0.038])  # sine
    popt, pcov = curve_fit(sine, selected_freqs / GHz, selected_mod_points, p0=p0)

    max_thickness = 0.5 * c0 / (2.7 * (pi / popt[1]) * GHz)
    if max(p) > max_thickness:
        return amp_loss*(max_thickness - max(p))**2#*p_loss
    else:
        return amp_loss
        """

    return amp_loss * p_loss


def rp(p):
    amplitude_sim = get_amplitude(freqs, p, n)
    enum = np.sum((amplitude_measured - np.mean(amplitude_measured)) * (amplitude_sim - np.mean(amplitude_sim)))
    denum1 = 1 / np.sqrt(np.sum((amplitude_measured - np.mean(amplitude_measured)) ** 2))
    denum2 = 1 / np.sqrt(np.sum((amplitude_sim - np.mean(amplitude_sim)) ** 2))

    return enum * denum1 * denum2


"""
n_min, n_max = 2.50, 3.15
n_mins, n_maxs = np.arange(n_min, n_max + 0.01, 0.01), np.arange(n_min, n_max + 0.01, 0.01)
p_opt = np.array([42.5, 641.3, 74.4]) * um_to_m
best_combo, min_val = (None, None), np.inf
for n1 in n_mins:
    for n2 in n_maxs:
        n = get_n(freqs, n1, n2)
        #t_loss = amplitude_loss(p_opt, n)*phase_loss(p_opt, n)
        #t_loss = phase_loss(p_opt, n)
        t_loss = amplitude_loss(p_opt, n)
        if t_loss < min_val:
            min_val = t_loss
            best_combo = (n1, n2)
            print(t_loss, best_combo)

print(freqs.min()/THz, freqs.max()/THz, np.mean(np.diff(freqs)/THz))
print(sam_idx, best_combo)
"""
if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(freqs / GHz, phase_measured, label="Phase measured", color="black")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Phase (rad)")

    ax2.plot(freqs / GHz, amplitude_measured, label="Amplitude measured", color="black")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Reflectance")
    fig.suptitle("Model vs measurement")

    # p = np.array([60, 620, 75]) * um_to_m
    p_opt = np.array([42.5, 641.3, 74.4]) * um_to_m
    p_opt_test = np.array([142.5, 541.3, 174.4]) * um_to_m
    p0 = np.array([45, 420, 75]) * um_to_m
    p1 = np.array([51, 608, 56]) * um_to_m
    p2 = np.array([26, 573, 91]) * um_to_m
    p4 = np.array([46, 147, 915]) * um_to_m
    p5 = np.array([42, 641, 157]) * um_to_m
    p6 = np.array([583, 980, 111]) * um_to_m
    p7 = np.array([41, 633, 924]) * um_to_m
    p8 = np.array([31, 307, 518]) * um_to_m
    p9 = np.array([31, 639, 71]) * um_to_m
    p10 = np.array([734, 1000, 458]) * um_to_m
    p11 = np.array([513, 739, 684]) * um_to_m
    p12 = np.array([31, 563, 71]) * um_to_m
    p13 = np.array([21, 895, 764]) * um_to_m
    p14 = np.array([252, 16, 849]) * um_to_m
    p15 = np.array([41, 634, 66]) * um_to_m
    p16 = np.array([201.8, 774.1, 909.6]) * um_to_m
    p17 = np.array([36.1, 623.5, 66.3]) * um_to_m
    p18 = np.array([36.1, 633.5, 66.3]) * um_to_m

    solutions = array([p_opt, p5, 100 * np.random.random(3) * um_to_m, p_opt_test])
    for p in solutions:
        phase_model = get_phase(freqs, p, n)
        amplitude_model = get_amplitude(freqs, p, n)
        t_loss = total_loss(p)
        print(p * 10 ** 6, t_loss)

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
        diff_phase, diff_amplitude = (phase_measured - phase_model) ** 2, (amplitude_measured - amplitude_model) ** 2

        plt_label_phase = f"Phase difference {p * 10 ** 6}, p_loss {round(float(phase_loss(p)), 6)}"
        ax1.plot(freqs / GHz, diff_phase, label=plt_label_phase)

        plt_label_amplitude = f"Amp. difference {p * 10 ** 6}, a_loss {round(float(amplitude_loss(p)), 6)}"
        ax2.plot(freqs / GHz, diff_amplitude, label=plt_label_amplitude)

    ax1.legend()
    ax2.legend()
    plt.show()

    print(f"Best solution: {best_sol * 10 ** 6}")
    print(f"Total_loss = loss_p*loss_a: {best_min_p}*{best_min_a}={best_min_p * best_min_a}")

    file_name = Path("image_files") / "interference_tests_6freq_grid_vals_v1_0_0_1"

    rez_x, rez_y, rez_z = 200, 200, 200
    lb = array([0.000001, 0.000001, 0.000001])
    ub = array([0.001000, 0.001000, 0.001000])
    # lb = array([0.000001, 0.000540, 0.000001])
    # ub = array([0.000200, 0.000740, 0.000200])

    new_settings = {"rez": (rez_x, rez_y, rez_z), "lb": lb, "ub": ub}

    try:
        grid_vals = np.load(str(file_name) + ".npy")
    except FileNotFoundError:
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
                    grid_vals[i, j, k] = total_loss(p)

        np.save(str(file_name), grid_vals)
    title = "Amp. loss"
    map_plot(img_data=grid_vals, representation="log", settings=new_settings, title=title)
