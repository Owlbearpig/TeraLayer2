import numpy as np
from numpy import cos, sin, arcsin, exp, dot, conj, pi
from consts import um_to_m, c0, THz, GHz, array
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit
from model.refractive_index import get_n, get_n_no_dispersion
from scipy.optimize import curve_fit
from model.measurement_data import get_measured_phase, get_measured_amplitude, get_ref_amplitude, get_ref_phase
from functions import noise_gen

mpl.rcParams['axes.grid'] = True
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['xtick.major.width'] = 2.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
# plt.style.use(['dark_background'])
# plt.xkcd()
mpl.rcParams.update({'font.size': 16})


# print(mpl.rcParams.keys())


@jit(cache=True, nopython=False)
def multir_complex(freqs, p, n):
    thea = 1 * 8.0 * pi / 180.0
    es = p.copy()

    the = np.zeros(len(freqs), dtype=np.complex128)
    ra, rb = np.zeros(len(freqs), dtype=np.complex128), np.zeros(len(freqs), dtype=np.complex128)
    ta, tb = np.zeros(len(freqs), dtype=np.complex128), np.zeros(len(freqs), dtype=np.complex128)
    the[0] = thea

    r = np.zeros(len(freqs), dtype=np.complex128)
    nc = 3

    for h in range(len(freqs)):
        for k in range(nc + 1):
            the[k + 1] = arcsin(n[h, k] * sin(the[k]) / n[h, k + 1])
            ra[k] = ((n[h, k] * cos(the[k + 1])) - ((n[h, k + 1]) * cos(the[k]))) / \
                    ((n[h, k + 1] * cos(the[k])) + (n[h, k] * cos(the[k + 1])))
            rb[k] = ((n[h, k + 1] * cos(the[k])) - (n[h, k] * cos(the[k + 1]))) / \
                    ((n[h, k] * cos(the[k + 1])) + (n[h, k + 1] * cos(the[k])))
            ta[k] = (2 * n[h, k] * cos(the[k + 1])) / \
                    ((n[h, k + 1] * cos(the[k])) + (n[h, k] * cos(the[k + 1])))
            tb[k] = (2 * n[h, k + 1] * cos(the[k])) / \
                    ((n[h, k] * cos(the[k + 1])) + (n[h, k + 1] * cos(the[k])))

        """ correct, line below with transmission coeff.
        M = (1 / tb[0]) * np.array([[(ta[0] * tb[0]) - (ra[0] * rb[0]), rb[0]],
                                    [-ra[0], 1]], dtype=np.complex128)
        """
        M = np.array([[(ta[0] * tb[0]) - (ra[0] * rb[0]), rb[0]],
                                    [-ra[0], 1]], dtype=np.complex128)
        fi = np.zeros(nc, dtype=np.complex128)
        #print(M)
        for s in range(nc):
            fi[s] = (2 * pi * n[h, s + 1] * es[s]) * (freqs[h] / c0)
            """ correct
            Q = (1 / tb[s + 1]) * np.array([[(ta[s + 1] * tb[s + 1]) - (ra[s + 1] * rb[s + 1]), rb[s + 1]],
                                            [-ra[s + 1], 1]], dtype=np.complex128)
            """
            Q = np.array([[(ta[s + 1] * tb[s + 1]) - (ra[s + 1] * rb[s + 1]), rb[s + 1]],
                                            [-ra[s + 1], 1]], dtype=np.complex128)

            P = np.array([[exp(-fi[s] * 1j), 0], [0, exp(fi[s] * 1j)]])
            M = dot(M, dot(P, Q))
        #print(h, M[0, 1])
        #print(h, M[1, 1])

        r[h] = M[0, 1] / M[1, 1]
        #print(r[h])

    #print(r)
    #print(np.conj(r)*r)
    #exit()

    return r


@jit(cache=True, nopython=True)
def tmm_matrix_elems(freqs, p, n):
    thea = 1 * 8.0 * pi / 180.0
    es = p.copy()

    the = np.zeros(len(freqs), dtype=np.complex128)
    ra, rb = np.zeros(len(freqs), dtype=np.complex128), np.zeros(len(freqs), dtype=np.complex128)
    ta, tb = np.zeros(len(freqs), dtype=np.complex128), np.zeros(len(freqs), dtype=np.complex128)
    the[0] = thea

    m01, m11 = np.zeros(len(freqs), dtype=np.complex128), np.zeros(len(freqs), dtype=np.complex128)
    r = np.zeros(len(freqs), dtype=np.complex128)
    nc = 3

    for h in range(len(freqs)):
        for k in range(nc + 1):
            the[k + 1] = arcsin(n[h, k] * sin(the[k]) / n[h, k + 1])
            ra[k] = ((n[h, k] * cos(the[k + 1])) - ((n[h, k + 1]) * cos(the[k]))) / \
                    ((n[h, k + 1] * cos(the[k])) + (n[h, k] * cos(the[k + 1])))
            rb[k] = ((n[h, k + 1] * cos(the[k])) - (n[h, k] * cos(the[k + 1]))) / \
                    ((n[h, k] * cos(the[k + 1])) + (n[h, k + 1] * cos(the[k])))
            ta[k] = (2 * n[h, k] * cos(the[k + 1])) / \
                    ((n[h, k + 1] * cos(the[k])) + (n[h, k] * cos(the[k + 1])))
            tb[k] = (2 * n[h, k + 1] * cos(the[k])) / \
                    ((n[h, k] * cos(the[k + 1])) + (n[h, k + 1] * cos(the[k])))

        """ correct, line below with transmission coeff.
        M = (1 / tb[0]) * np.array([[(ta[0] * tb[0]) - (ra[0] * rb[0]), rb[0]],
                                    [-ra[0], 1]], dtype=np.complex128)
        """
        #"""
        M = np.array([[(ta[0] * tb[0]) - (ra[0] * rb[0]), rb[0]],
                                    [-ra[0], 1]], dtype=np.complex128)
        #"""
        fi = np.zeros(nc, dtype=np.complex128)
        #print(M)
        for s in range(nc):
            fi[s] = (2 * pi * n[h, s + 1] * es[s]) * (freqs[h] / c0)
            """ correct
            Q = (1 / tb[s + 1]) * np.array([[(ta[s + 1] * tb[s + 1]) - (ra[s + 1] * rb[s + 1]), rb[s + 1]],
                                            [-ra[s + 1], 1]], dtype=np.complex128)
            """
            #"""
            Q = np.array([[(ta[s + 1] * tb[s + 1]) - (ra[s + 1] * rb[s + 1]), rb[s + 1]],
                                            [-ra[s + 1], 1]], dtype=np.complex128)
            #"""
            P = np.array([[exp(-fi[s] * 1j), 0], [0, exp(fi[s] * 1j)]])
            M = dot(M, dot(P, Q))

        m01[h], m11[h] = M[0, 1], M[1, 1]
        #r[h] = M[0, 1] / M[1, 1]

    return m01, m11 # r


@jit(cache=True, nopython=True)
def custom_unwrap(phase):
    p = phase.copy()
    for i in range(1, len(phase) - 1):
        diff = phase[i - 1] - phase[i]
        if np.abs(diff) > pi * 0.9:
            p[i:] += diff

    return p


@jit(cache=True, nopython=False)
def get_phase(freqs, p, n):
    r = multir_complex(freqs, p, n)

    return np.angle(r)


@jit(cache=True, nopython=False)
def get_amplitude(freqs, p, n):
    r = multir_complex(freqs, p, n)

    return np.real(r * conj(r))


@jit(cache=True, nopython=False)
def get_r_cart(freqs, p, n):
    r = multir_complex(freqs, p, n)

    return r


def unwrap(phase):
    p_uwrapped = phase.copy()
    for i in range(1, len(phase) - 1):
        diff = phase[i - 1] - phase[i]
        if np.abs(diff) > 1:
            p_uwrapped[i:] += np.sign(diff) * pi

    return p_uwrapped


def thickest_layer_approximation(freqs, model_data):
    # assuming first two points are below the first interference minima
    def sine(x, a, omega):
        return a ** 2 * np.sin(x * omega) ** 2

    p0 = np.array([np.sqrt(0.537), 0.038])  # sine
    popt, pcov = curve_fit(sine, freqs[:2] / GHz, model_data[:2], p0=p0)
    n = 2.7
    thickest_layer = 0.95 * c0 / (2 * n * (pi / popt[1]) * GHz)  # works pretty well with 0.95
    print(popt)
    print("estimated max(p): ", 10 ** 6 * thickest_layer)

    return thickest_layer


if __name__ == '__main__':
    # freqs = array([0.400, 0.480, 0.560, 0.640, 0.720, 0.800]) * THz
    all_freqs = np.arange(0.001, 1.400 + 0.001, 0.001) * THz
    freqs = array([0.040, 0.080, 0.150, 0.550, 0.640, 0.760]) * THz
    freqs = all_freqs.copy()
    n = get_n(freqs, 2.80, 2.80)

    # n = get_n_no_dispersion(freqs, 2.70)

    p_opt = np.array([42.5, 641.3, 74.4]) * um_to_m

    sam_idx = 28
    phase_measured = get_measured_phase(freqs, sam_idx)
    amplitude_measured = get_measured_amplitude(freqs, sam_idx)

    limited_slice = np.abs(phase_measured) <= pi
    phase_measured = phase_measured[limited_slice]
    amplitude_measured = amplitude_measured[limited_slice]
    freqs_filtered = freqs[limited_slice]

    noise_std_scale = 0.50
    noise_amp = noise_gen(freqs, True, scale=0.15*noise_std_scale)
    noise_phase = noise_gen(freqs, True, scale=0.10*noise_std_scale)
    phase_mod = get_phase(freqs, p_opt, n) + noise_phase
    amp_mod = get_amplitude(freqs, p_opt, n) * (1 + noise_amp) ** 2

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(freqs / GHz, phase_mod, label=f"Noisy phase model, {p_opt * 10 ** 6}")
    ax1.plot(freqs_filtered / GHz, phase_measured, label=f"Phase measured, sam_idx: {sam_idx}")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Phase (rad)")
    ax1.legend()

    ax2.plot(freqs / GHz, amp_mod, label=f"Noisy intensity model, {p_opt * 10 ** 6}")
    ax2.plot(freqs_filtered / GHz, amplitude_measured, label=f"Int. measured, sam_idx: {sam_idx}")
    selected_freqs = array([0.040, 0.080, 0.150, 0.550, 0.640, 0.760]) * 1000
    for xc in selected_freqs:
        continue
        ax2.axvline(x=xc, color="red")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Int. (a.u.)")
    ax2.legend()
    plt.show()
