import numpy as np
from numpy import cos, sin, arcsin, exp, dot, conj, pi
from consts import um_to_m, c0, THz, GHz
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit
from model.refractive_index import get_n
from measurement_data import get_measured_phase, get_measured_amplitude

mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['axes.grid'] = True

# print(mpl.rcParams.keys())


@jit(cache=True, nopython=True)
def multir_complex(freqs, p, n):
    thea = 8.0 * pi / 180.0
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


if __name__ == '__main__':
    from consts import array

    #freqs = array([0.400, 0.480, 0.560, 0.640, 0.720, 0.800]) * THz
    freqs = np.arange(0.250, 1.400 + 0.001, 0.001) * THz
    n = get_n(freqs, 3.00, 3.00)
    #p_opt = np.array([42.5, 641.3, 74.4]) * um_to_m
    p_opt = np.array([42.5/2, 641.3, 74.4/2]) * um_to_m

    sam_idx = 78
    phase_measured = get_measured_phase(freqs, sam_idx)

    limited_slice = np.abs(phase_measured) <= pi
    phase_measured = phase_measured[limited_slice]
    #freqs = freqs[limited_slice]

    phase_mod = get_phase(freqs, p_opt, n)

    plt.plot(freqs / GHz, phase_mod, label="phase model")
    #plt.plot(freqs / GHz, phase_measured, label="phase measured")
    plt.xlabel("frequency (GHz)")
    plt.ylabel("phase (rad)")
    plt.legend()
    plt.show()
