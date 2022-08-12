import numpy as np
from numpy import cos, sin, arcsin, exp, dot, conj, pi
from consts import um_to_m, c0
from model.multir_numba import multir_numba
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['axes.grid'] = True
# print(mpl.rcParams.keys())

def multir_complex(freqs, p, n):
    thea = 8.0 * pi / 180.0
    es = p.copy()

    the = np.zeros(len(freqs), dtype=np.complex128)
    ra, rb = np.zeros(len(freqs), dtype=np.complex128), np.zeros(len(freqs), dtype=np.complex128)
    ta, tb = np.zeros(len(freqs), dtype=np.complex128), np.zeros(len(freqs), dtype=np.complex128)
    the[0] = thea * 0

    r = np.zeros(len(freqs), dtype=np.complex128)
    nc = 3
    for h in range(len(freqs)):
        for k in range(nc + 1):
            the[k + 1] = arcsin(n[k] * sin(the[k]) / n[k + 1])
            ra[k] = ((n[k] * cos(the[k + 1])) - ((n[k + 1]) * cos(the[k]))) / \
                    ((n[k + 1] * cos(the[k])) + (n[k] * cos(the[k + 1])))
            rb[k] = ((n[k + 1] * cos(the[k])) - (n[k] * cos(the[k + 1]))) / \
                    ((n[k] * cos(the[k + 1])) + (n[k + 1] * cos(the[k])))
            ta[k] = (2 * n[k] * cos(the[k + 1])) / \
                    ((n[k + 1] * cos(the[k])) + (n[k] * cos(the[k + 1])))
            tb[k] = (2 * n[k + 1] * cos(the[k])) / \
                    ((n[k] * cos(the[k + 1])) + (n[k + 1] * cos(the[k])))

        M = (1 / tb[0]) * np.array([[(ta[0] * tb[0]) - (ra[0] * rb[0]), rb[0]],
                                    [-ra[0], 1]], dtype=np.complex128)

        fi = np.zeros(nc, dtype=np.complex128)
        for s in reversed(range(nc)):
            fi[s] = (2 * pi * n[s + 1] * es[s]) * (freqs[h] * 10 ** 12 / c0)
            Q = (1 / tb[s + 1]) * np.array([[(ta[s + 1] * tb[s + 1]) - (ra[s + 1] * rb[s + 1]), rb[s + 1]],
                                            [-ra[s + 1], 1]], dtype=np.complex128)
            P = np.array([[exp(-fi[s] * 1j), 0], [0, exp(fi[s] * 1j)]])
            M = dot(dot(Q, P), M)

        r[h] = M[0, 1] / M[1, 1]

    return r


if __name__ == '__main__':
    freqs = np.arange(0.02, 2.00, 0.001)

    #p = np.array([55, 640, 75]) * um_to_m
    p = np.array([150, 100, 200])*um_to_m
    n = np.array([1.00 + 0j, 1.50 + 0j, 1.80 + 0j, 1.20 + 0j, 1.00 + 0j], dtype=np.complex128)

    R_C = multir_complex(freqs, p, n)

    plt.figure("Amplitude frequency domain (Sim TMM)")
    plt.plot(freqs, 20 * np.log10(np.abs(R_C)), label="Sam/Ref")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Reflectance (dB)")
    plt.legend()

    plt.figure("Raw phase frequency domain (Sim TMM)")
    # p_uwrap_sam = np.arctan2(Y2[idx].imag, Y2[idx].real)
    # p_uwrap_ref = np.arctan2(Y[idx].imag, Y[idx].real)
    p_unwrap_diff = np.arctan2(R_C.imag, R_C.real)
    plt.plot(freqs, p_unwrap_diff, label="Sam - Ref")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (rad)")
    plt.legend()

    plt.figure("Unwrapped phase frequency domain (Sim TMM)")
    p_uwrap = np.unwrap(p_unwrap_diff)
    # plt.plot(freq[idx], p_uwrap_ref, label="Reference")
    # plt.plot(freq[idx], p_uwrap_sam, label="Sample")
    plt.plot(freqs, p_uwrap, label="Sam - Ref")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (rad)")
    plt.legend()

    plt.show()

    print(f"R_C: {R_C}")
    print(f"Sum R complex version: {np.sum(R_C)}")
