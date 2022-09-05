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
        freqs = array([0.440, 0.520, 0.600, 0.640, 0.780, 0.860]) * THz
        freqs = array([420, 440, 480, 520, 560, 600], dtype=float) * GHz
        freqs = array([430, 460, 490, 520, 550, 590], dtype=float) * GHz

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
    n = get_n(freqs, 2.70, 2.70)

    phase_sim = get_phase(freqs, p, n)
    amplitude_sim = get_amplitude(freqs, p, n)

    p_loss = np.sum((phase_sim - phase_measured) ** 2)
    amp_loss = np.sum((amplitude_sim - amplitude_measured) ** 2)

    return amp_loss * p_loss


if __name__ == '__main__':
    from scipy.optimize import minimize

    p_opt = np.array([42.5, 641.3, 74.4]) * um_to_m

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
    exit()

    sam_idx = 78
    p0 = p_opt.copy()

    t_loss = calc_loss(p_opt, sam_idx)
    print(p_opt * 10 ** 6, t_loss)

    p_minima = []
    for sam_idx in np.arange(0, 101):
        res = minimize(calc_loss, p_opt, args=[sam_idx], method='Nelder-Mead')
        p_minima.append(res.x)
        # print(res.x / um_to_m)
        # print(res)

    p_minima = array(p_minima)

    print(np.mean(p_minima[:, 0] / um_to_m), np.mean(p_minima[:, 1] / um_to_m), np.mean(p_minima[:, 2] / um_to_m))
    print(np.std(p_minima[:, 0] / um_to_m), np.std(p_minima[:, 1] / um_to_m), np.std(p_minima[:, 2] / um_to_m))

    plt.plot(np.arange(0, 101), p_minima[:, 0] / um_to_m, label="d1")
    plt.plot(np.arange(0, 101), p_minima[:, 1] / um_to_m, label="d2")
    plt.plot(np.arange(0, 101), p_minima[:, 2] / um_to_m, label="d3")
    plt.legend()
    plt.xlabel("sample idx")
    plt.ylabel("thickness (um)")
    plt.show()
