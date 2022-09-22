import numpy as np
from numpy import cos, sin, arcsin, exp, dot, conj, pi
from consts import um_to_m, c0, THz, array
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit
from measurement_data import get_measured_phase, get_measured_amplitude
from refractive_index import get_n

mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['axes.grid'] = True

nc = 3  # Note: This is only for 3 layers.


class TmmFast:
    def __init__(self, freqs):
        self.freqs = freqs

        self.n = get_n(self.freqs, 2.70, 2.70)

        self.the = np.zeros(len(self.freqs), float)
        self.the[0] = 0 * 8.0 * pi / 180.0

    def interface_coefficients(self):
        ra, rb = np.zeros((len(freqs), nc + 1), float), np.zeros((len(freqs), nc + 1), float)
        ta, tb = np.zeros((len(freqs), nc + 1), float), np.zeros((len(freqs), nc + 1), float)
        for h in range(len(freqs)):
            for k in range(nc + 1):
                self.the[k + 1] = arcsin(self.n[h, k] * sin(self.the[k]) / self.n[h, k + 1])
                ra[h, k] = ((self.n[h, k] * cos(self.the[k + 1])) - ((self.n[h, k + 1]) * cos(self.the[k]))) / \
                           ((self.n[h, k + 1] * cos(self.the[k])) + (self.n[h, k] * cos(self.the[k + 1])))
                rb[h, k] = ((self.n[h, k + 1] * cos(self.the[k])) - (self.n[h, k] * cos(self.the[k + 1]))) / \
                           ((self.n[h, k] * cos(self.the[k + 1])) + (self.n[h, k + 1] * cos(self.the[k])))
                ta[h, k] = (2 * self.n[h, k] * cos(self.the[k + 1])) / \
                           ((self.n[h, k + 1] * cos(self.the[k])) + (self.n[h, k] * cos(self.the[k + 1])))
                tb[h, k] = (2 * self.n[h, k + 1] * cos(self.the[k])) / \
                           ((self.n[h, k] * cos(self.the[k + 1])) + (self.n[h, k + 1] * cos(self.the[k])))
        return ra, rb, ta, tb

    def phase_prefactors(self):
        f1, f2 = np.zeros(len(self.freqs), float), np.zeros(len(self.freqs), float)
        for i in range(len(self.freqs)):
            f1[i] = 2 * pi * self.n[i, 1] * (self.freqs[i] / c0)
            f2[i] = 2 * pi * self.n[i, 2] * (self.freqs[i] / c0)

        return f1, f2

    def reflectivity(self, p):
        ra, rb, _, _ = self.interface_coefficients()
        a, b = 0.2, 0.28571429

        def c_mod(s):
            res = s - 2 * pi * (int(s / (2 * pi)) - (s < 0)) - pi
            return res

        def sine(x):
            B = 4 / pi
            C = -4 / (pi * pi)

            y = x * (B + C * abs(x))

            P = 0.225
            res = P * y * (abs(y) - 1) + y

            return res

        def cose(x):
            x += pi / 2
            x -= (x > pi) * (2 * pi)

            return sine(x)

        f, g = self.phase_prefactors()
        r = np.zeros(len(self.freqs), dtype=np.complex128)

        for i in range(len(self.freqs)):
            f1 = f[i] * p[0]
            f2 = g[i] * p[1]
            f3 = f[i] * p[2]

            m01_0 = (exp(1j * (-f1 - f2)) + b * a * exp(1j * (f1 - f2))) * (-a * exp(1j * (-f3)) - b * exp(1j * f3))
            m01_1 = (b * exp(1j * (f2 - f1)) + a * exp(1j * (f1 + f2))) * (a * b * exp(1j * (-f3)) + exp(1j * f3))

            m01 = m01_0 + m01_1

            m11_0 = (a * exp(1j * (-f1 - f2)) + b * exp(1j * (-f2 + f1))) * (-a * exp(1j * (-f3)) - b * exp(1j * f3))
            m11_1 = (a * b * exp(1j * (-f1 + f2)) + exp(1j * (f1 + f2))) * (a * b * exp(1j * (-f3)) + exp(1j * f3))

            m11 = m11_0 + m11_1
            """
            m01_0 = -a[i] * exp(1j * (-f0 - f1 - f2))
            m01_1 = -b[i] * exp(1j * (-f0 - f1 + f2))
            m01_2 = +a[i] ** 2 * b[i] * exp(1j * (f0 - f1 - f2))
            m01_3 = +a[i] * b[i] ** 2 * exp(1j * (f0 - f1 - f2))
            m01_4 = -a[i] * b[i] ** 2 * exp(1j * (-f0 + f1 - f2))
            m01_5 = +b[i] * exp(1j * (f0 - f1 - f2))
            m01_6 = -a[i] ** 2 * b[i] * exp(1j * (-f0 - f1 + f2))
            m01_7 = +a[i] * exp(1j * (f0 + f1 + f2))

            m11_0 = +a[i] ** 2 * exp(1j * (-f0 - f1 - f2))
            m11_1 = +a[i] * b[i] * exp(1j * (f2 - f0 - f1))
            m11_2 = +a[i] * b[i] * exp(1j * (f0 - f1 - f2))
            m11_3 = +b[i] ** 2 * exp(1j * (f0 - f1 + f2))
            m11_4 = +b[i] ** 2 * a[i] ** 2 * exp(1j * (-f0 + f1 - f2))
            m11_5 = -a[i] * b[i] * exp(1j * (f0 + f1 - f2))
            m11_6 = -a[i] * b[i] * exp(1j * (-f0 + f1 + f2))
            m11_7 = +exp(1j * (f0 + f1 + f2))

            m01 = m01_0 + m01_1 + m01_2 + m01_3 + m01_4 + m01_5 + m01_6 + m01_7
            m11 = m11_0 + m11_1 + m11_2 + m11_3 + m11_4 + m11_5 + m11_6 + m11_7
            """
            r[i] = m01 / m11
            """
            s0, s1, s2, s3 = f2 + f1 + f0, f2 - f1 - f0, f2 + f1 - f0, - f2 + f1 - f0



            # print("s0, s1, s2, s3", s0, s1, s2, s3)
            s0 = c_mod(s0)
            s1 = c_mod(s1)
            s2 = c_mod(s2)
            s3 = c_mod(s3)
            # print("s0, s1, s2, s3", s0, s1, s2, s3)
            ss0, ss1, ss2, ss3 = sine(s0), sine(s1), sine(s2), sine(s3)
            cs0, cs1, cs2, cs3 = cose(s0), cose(s1), cose(s2), cose(s3)

            # print("ss0, ss1, ss2, ss3", ss0, ss1, ss2, ss3)
            # print("cs0, cs1, cs2, cs3", cs0, cs1, cs2, cs3)

            m_12_r = (1 - a[i] * a[i]) * b[i] * (cs2 - cs1)
            m_22_r = (1 - a[i] * a[i]) * (cs0 - b[i] * b[i] * cs3)

            m_12_i = - 2 * a[i] * (ss0 + b[i] * b[i] * ss3) + (a[i] * a[i] + 1) * b[i] * (ss1 - ss2)
            m_22_i = (a[i] * a[i] + 1) * (ss0 + b[i] * b[i] * ss3) + 2 * a[i] * b[i] * (ss2 - ss1)
            
            # e = (m_12_r * m_12_r + m_12_i * m_12_i)
            # d = (m_22_r * m_22_r + m_22_i * m_22_i)
            # print("m_12_r, m_12_i, m_22_r, m_22_i", m_12_r, m_12_i, m_22_r, m_22_i)
            # print(f"d freq. idx: {i}: {d}")
            # print(f"1/d freq. idx: {i}: {1/d}")

            r[i] = (m_12_r + 1j * m_12_i) / (m_22_r + 1j * m_22_i)
            """
        return r


if __name__ == '__main__':
    from tmm import multir_complex
    from timeit import default_timer

    freqs = array([0.040, 0.080, 0.150, 0.550, 0.640, 0.760]) * THz

    new_calc = TmmFast(freqs)

    p_opt = array([90, 850, 110]) * um_to_m

    r = new_calc.reflectivity(p_opt)
    print(r)
    r_correct = multir_complex(freqs, p_opt, new_calc.n)
    print(r_correct)

    """
    #new_calc.calc_total_loss(p_opt * um_to_m)
    t0 = default_timer()
    for p in np.random.random((1000, 6)):
        new_calc.calc_total_loss(p * um_to_m)

    print(default_timer() - t0)
    """
