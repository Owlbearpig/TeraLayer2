import numpy as np
import matplotlib.pyplot as plt
from model.tmm import get_phase, get_amplitude
from measurement_data import get_measured_phase, get_measured_amplitude
from consts import um_to_m, THz, array, GHz, pi, ones
from visualizing.simplecolormap import map_plot
from refractive_index import get_n
from random import randint


def bad_ones(sam_idx):
    freqs = array([400, 440, 480, 520, 560, 600], dtype=float) * GHz

    phase_measured = get_measured_phase(freqs, sam_idx)
    amplitude_measured = get_measured_amplitude(freqs, sam_idx)

    limited_slice = np.abs(phase_measured) <= pi
    phase_measured = phase_measured[limited_slice]
    amplitude_measured = amplitude_measured[limited_slice]
    freqs = freqs[limited_slice]

    if (len(phase_measured) < 6):
        print(freqs / THz)
        print(len(phase_measured))
        print(sam_idx)


def calc_loss(p, sam_idx, freqs=None):
    if freqs is None:
        # freqs = array([0.440, 0.520, 0.600, 0.640, 0.780, 0.860]) * THz
        # freqs = array([0.250, 0.350, 0.440, 0.520, 0.640, 0.860]) * THz
        # freqs = array([0.460, 0.490, 0.600, 0.640, 0.780, 0.840]) * THz
        # freqs = array([420, 440, 780, 860, 908, 938], dtype=float) * GHz
        # freqs = array([430, 460, 490, 520, 550, 590], dtype=float) * GHz
        # freqs = array([8.480e+11, 8.780e+11, 9.080e+11, 9.380e+11, 9.680e+11, 1.008e+12])
        # freqs = array([5.81e+11, 8.780e+11, 9.080e+11, 9.380e+11, 9.680e+11, 1.008e+12])
        # freqs = array([7.66e+11, 7.96e+11, 8.26e+11, 8.56e+11, 8.86e+11, 9.26e+11]) + 10*GHz # this shows point of t_loss vs amp_loss only
        freqs = array([0.460, 0.490, 0.600, 0.640, 0.780, 0.840]) * THz

    phase_measured = get_measured_phase(freqs, sam_idx)
    amplitude_measured = get_measured_amplitude(freqs, sam_idx)

    limited_slice = np.abs(phase_measured) <= pi
    phase_measured = phase_measured[limited_slice]
    amplitude_measured = amplitude_measured[limited_slice]
    freqs = freqs[limited_slice]

    """
    if (len(phase_measured) < 6):
        print(freqs / THz)
        print(len(phase_measured))
        print(sam_idx)
        # exit()
    """

    # n = get_n(freqs, 2.70, 2.85)
    n = get_n(freqs, 2.70, 2.75)

    phase_measured = get_phase(freqs, np.array([42.5, 641.3, 74.4]) * um_to_m, n)
    amplitude_measured = get_amplitude(freqs, np.array([42.5, 641.3, 74.4]) * um_to_m, n)

    phase_sim = get_phase(freqs, p, n)
    amplitude_sim = get_amplitude(freqs, p, n)

    # "original"
    """
    p_loss = np.sum((phase_sim - phase_measured) ** 2)
    amp_loss = np.sum((amplitude_sim - amplitude_measured) ** 2)
    return amp_loss*p_loss * (np.sum(p) - np.sum(p_opt))**2
    """
    # testing
    p_loss = np.sum((1 / len(freqs)) * ((phase_sim - phase_measured) / (2 * pi)) ** 2)
    amp_loss = np.sum((1 / len(freqs)) * (amplitude_sim - amplitude_measured) ** 2)

    # loss = amp_loss * p_loss + ((np.sum(p) - np.sum(p_opt)) * (1 / 3e-3)) ** 2# + amp_loss * p_loss
    loss = amp_loss * p_loss + ((np.sum(p) - np.sum(p_opt)) * (1 / 3e-3)) ** 2  # + amp_loss * p_loss

    return -np.log10(1 / loss)


if __name__ == '__main__':
    from scipy.optimize import minimize

    np.random.seed(420)

    p_opt = np.array([42.5, 641.3, 74.4]) * um_to_m

    """
    freqs = array([430, 460, 490, 520, 550, 590], dtype=np.float64) * GHz
    
    best_freq_set, min_val = None, np.inf
    for freq_offset in np.arange(0, 1000, 10):
        print(freq_offset)
        freqs = freqs + (freq_offset * GHz)

        p_minima = []
        for sam_idx in np.arange(0, 101):
            res = minimize(calc_loss, p_opt, args=(sam_idx, freqs), method='Nelder-Mead')
            p_minima.append(res.x)
        p_minima = array(p_minima)

        std = np.sum(np.std(p_minima, axis=0))
        print(std, freqs)
        if std < min_val:
            min_val = std
            best_freq_set = freqs.copy()

    print(min_val)
    print(best_freq_set)
    """

    sam_idx = 78
    p0 = p_opt * (0.9 + np.random.random(3) / 10)

    t_loss = calc_loss(p_opt, sam_idx)
    print("p0:", p0 * 10 ** 6, calc_loss(p0, sam_idx))
    print("p_opt:", p_opt * 10 ** 6, t_loss)

    sam_range = np.arange(0, 5)
    p_minima = []
    for sam_idx in sam_range:
        res = minimize(calc_loss, p0, args=[sam_idx], method='Nelder-Mead')
        p_minima.append(res.x)
        # print(res.x / um_to_m)
        # print(res)

    p_minima = array(p_minima)

    print(np.mean(p_minima[:, 0] / um_to_m), np.mean(p_minima[:, 1] / um_to_m), np.mean(p_minima[:, 2] / um_to_m))
    print(np.std(p_minima[:, 0] / um_to_m), np.std(p_minima[:, 1] / um_to_m), np.std(p_minima[:, 2] / um_to_m))

    plt.plot(sam_range, p_minima[:, 0] / um_to_m, label="d1")
    plt.plot(sam_range, p_minima[:, 1] / um_to_m, label="d2")
    plt.plot(sam_range, p_minima[:, 2] / um_to_m, label="d3")
    plt.legend()
    plt.xlabel("sample idx")
    plt.ylabel("thickness (um)")
    plt.show()
