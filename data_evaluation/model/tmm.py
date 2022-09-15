import numpy as np
from numpy import cos, sin, arcsin, exp, dot, conj, pi
from consts import um_to_m, c0, THz, GHz
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit
from model.refractive_index import get_n
from scipy.optimize import curve_fit
from model.measurement_data import get_measured_phase, get_measured_amplitude

mpl.rcParams['axes.grid'] = True
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['xtick.major.width'] = 2.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
#plt.style.use(['dark_background'])
#plt.xkcd()
mpl.rcParams.update({'font.size': 22})

# print(mpl.rcParams.keys())


@jit(cache=True, nopython=True)
def multir_complex(freqs, p, n):
    thea = 0 * 8.0 * pi / 180.0
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

        M = (1 / tb[0]) * np.array([[(ta[0] * tb[0]) - (ra[0] * rb[0]), rb[0]],
                                    [-ra[0], 1]], dtype=np.complex128)

        fi = np.zeros(nc, dtype=np.complex128)
        for s in range(nc):
            fi[s] = (2 * pi * n[h, s + 1] * es[s]) * (freqs[h] / c0)
            Q = (1 / tb[s + 1]) * np.array([[(ta[s + 1] * tb[s + 1]) - (ra[s + 1] * rb[s + 1]), rb[s + 1]],
                                            [-ra[s + 1], 1]], dtype=np.complex128)
            P = np.array([[exp(-fi[s] * 1j), 0], [0, exp(fi[s] * 1j)]])
            M = dot(M, dot(P, Q))

        r[h] = M[0, 1] / M[1, 1]

    return r


# @jit(cache=True, nopython=True)
def custom_unwrap(phase):
    p = phase.copy()
    for i in range(1, len(phase) - 1):
        diff = phase[i - 1] - phase[i]
        if np.abs(diff) > pi * 0.9:
            p[i:] += diff

    return p


@jit(cache=True, nopython=False)
def get_phase(freqs, p, n):
    R_C = multir_complex(freqs, p, n)

    return np.angle(R_C)


@jit(cache=True, nopython=False)
def get_amplitude(freqs, p, n):
    r_c = multir_complex(freqs, p, n)

    return np.real(r_c * conj(r_c))


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
        return a**2 * np.sin(x * omega)**2

    p0 = np.array([np.sqrt(0.537), 0.038])  # sine
    popt, pcov = curve_fit(sine, freqs[:2] / GHz, model_data[:2], p0=p0)
    n = 2.7
    thickest_layer = 0.95 * c0 / (2 * n * (pi / popt[1]) * GHz) # works pretty well with 0.95
    print(popt)
    print("estimated max(p): ", 10 ** 6 * thickest_layer)

    return thickest_layer


if __name__ == '__main__':
    from consts import array

    # freqs = array([0.400, 0.480, 0.560, 0.640, 0.720, 0.800]) * THz
    all_freqs = np.arange(0.001, 1.400 + 0.001, 0.001) * THz

    freqs = all_freqs.copy()
    n = get_n(freqs, 2.70, 2.70)
    print(n[0, :])
    # p_opt = np.array([42.5, 641.3, 74.4]) * um_to_m
    #p_opt = np.array([142.5, 541.3, 174.4]) * um_to_m
    p_opt = array([400, 680, 125.]) * um_to_m
    # p_opt = np.array([20, 350, 120]) * um_to_m

    sam_idx = 78
    # phase_measured = get_measured_phase(freqs, sam_idx)

    # limited_slice = np.abs(phase_measured) <= pi
    # phase_measured = phase_measured[limited_slice]
    # freqs = freqs[limited_slice]

    phase_mod = get_phase(freqs, p_opt, n)
    amp_mod = get_amplitude(freqs, p_opt, n)
    slope_slice = (freqs < 1400 * GHz) * (freqs >= 1 * GHz)
    print(sum(slope_slice))
    print(np.mean(np.diff(unwrap(phase_mod))))

    plt.figure()
    plt.plot(freqs[:-1] / GHz, np.diff(unwrap(phase_mod)), label=f"{p_opt * 10 ** 6}")
    plt.xlabel("frequency (GHz)")
    plt.ylabel("phase diff (rad)")
    plt.legend()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(freqs / GHz, np.unwrap(phase_mod), label=f"phase model, {p_opt * 10 ** 6}")
    # ax1.plot(freqs / GHz, phase_measured, label="phase measured")
    ax1.set_xlabel("frequency (GHz)")
    ax1.set_ylabel("phase (rad)")
    ax1.legend()

    ax2.plot(freqs / GHz, amp_mod, label=f"amplitude model, {p_opt * 10 ** 6}")
    selected_freqs = array([0.040, 0.080, 0.150, 0.550, 0.640, 0.760]) * 1000
    selected_freqs = array([0.020, 0.060, 0.150, 0.550, 0.640, 0.760]) * 1000
    for xc in selected_freqs:
        ax2.axvline(x=xc, color="red")
    ax2.set_xlabel("frequency (GHz)")
    ax2.set_ylabel("amp (a.u.)")
    ax2.legend()

    print(amp_mod[10])

    plt.figure()


    def sine(x, a, omega):
        return a * np.sin(x * omega)**2


    fit_slice = (0 * GHz < all_freqs) * (all_freqs < 150 * GHz)
    selected_freqs = selected_freqs[:2] * GHz
    selected_mod_points = get_amplitude(selected_freqs, p_opt, n)

    p0 = array([0.554, 0.035])  # sine
    popt, pcov = curve_fit(sine, selected_freqs / GHz, selected_mod_points, p0=p0)
    print(popt)
    print(pi / popt[1])
    print(10 ** 6 * c0 / (2 * 2.7 * (pi / popt[1]) * GHz))
    thickness = 10 ** 6 * c0 / (2 * 2.7 * (pi / popt[1]) * GHz)
    plt.title(r"Fit of A$\sin(\nu\omega)^2$")
    plt.text(110, 0.15, fr"$\pi/\omega={round(pi/popt[1],2)}$ GHz")
    plt.text(110, 0.05, r"$\frac{mc}{2n\omega}=$")
    plt.text(120, 0.05, rf"{round(thickness, 1)} ($\mu$m)")
    plt.plot(all_freqs[fit_slice] / GHz, amp_mod[fit_slice], label=f"Model {p_opt*10**6} ($\mu$m)")
    plt.scatter(selected_freqs / GHz, selected_mod_points, color="red", label="Model at selected frequencies", s=40)
    plt.plot(all_freqs[fit_slice] / GHz, sine(all_freqs[fit_slice] / GHz, *popt), label="Sine fit")
    #plt.plot(all_freqs[fit_slice] / GHz, sine(all_freqs[fit_slice] / GHz, *p0), label="sine fit at p0")

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend(loc='upper right')
    plt.show()
