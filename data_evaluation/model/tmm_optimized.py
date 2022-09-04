import numpy as np
from numpy import cos, sin, arcsin, exp, dot, conj, pi
from consts import um_to_m, c0, THz
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit
from measurement_data import get_measured_phase, get_measured_amplitude
from refractive_index import get_n

mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['axes.grid'] = True

# print(mpl.rcParams.keys())


class TmmFast:
    def __init__(self, freqs, sam_idx):
        phase_measured = get_measured_phase(freqs, sam_idx)
        amplitude_measured = get_measured_amplitude(freqs, sam_idx)

        limited_slice = np.abs(phase_measured) <= pi
        self.phase_measured = phase_measured[limited_slice]
        self.amplitude_measured = amplitude_measured[limited_slice]

        self.freqs = freqs[limited_slice]

        self.n = get_n(self.freqs, 2.70, 2.85)

    def calc_total_loss(self, p):
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

        r_c = multir_complex(self.freqs, p, self.n)
        mod_p, mod_a = np.angle(r_c), np.real(r_c * conj(r_c))

        loss = np.sum((mod_a - self.amplitude_measured) ** 2) * np.sum((mod_p - self.phase_measured) ** 2)

        return loss



if __name__ == '__main__':
    from timeit import default_timer
    sam_idx = 78
    freqs = np.arange(0.400, 1.400 + 0.001, 0.001) * THz

    new_calc = TmmFast(freqs, sam_idx)

    p_opt = np.array([42.5, 641.3, 74.4]) * um_to_m
    p0 = np.array([45, 420, 75]) * um_to_m
    new_calc.calc_total_loss(p_opt * um_to_m)

    t0 = default_timer()
    for p in np.random.random((1000, 6)):
        new_calc.calc_total_loss(p*um_to_m)

    print(default_timer() - t0)


