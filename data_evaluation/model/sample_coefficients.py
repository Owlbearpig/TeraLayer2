import numpy as np
from tmm import (is_forward_angle, list_snell, seterr, interface_t, interface_r,
                 make_2x2_array)
from functools import partial
import sys
from numpy import array, pi, cos
from scipy.constants import c as c0
from consts import selected_freqs
from numfi import numfi as numfi_
from meas_eval.cw.refractive_index_fit import freqs as all_freqs

EPSILON = sys.float_info.epsilon  # typical floating-point calculation error

""" notes
1. snell(n, th)
2. phases(snell, n, lam)
3. delta = phases * d
4. t_list(n, snell) (interfaces)
5. exp(delta)
6. M = (Pi * Ri)*(Pi * Ri)*(Pi * Ri)
7. r = M(1,0) / M(0,0)

1. snell(n, th)
2. phases(snell, n, lam)
3. t_list(n, snell)

4. delta = phases * d
5. Pi = exp(-1j*delta[i]), 0, 0, exp(1j*delta[i])
6. M = (Pi * Ri)*(Pi * Ri)*(Pi * Ri)
7. r = M(1,0) / M(0,0)

d_list
[inf, 43.0, 641.0, 74.0, inf]

kz_list
[0.00871688 0.01367722 0.02532172 0.01367722 0.00871688]
[0.01079233 0.01704312 0.03145982 0.01704312 0.01079233]
[0.01349041 0.02171417 0.03932477 0.02171417 0.01349041]
[0.01660359 0.02689344 0.04856758 0.02689344 0.01660359]
[0.01764131 0.02875309 0.05124635 0.02875309 0.01764131]
[0.01971676 0.03213581 0.05747466 0.03213581 0.01971676]

delta = kz_list * d
[        inf  0.58812049 16.23122493  1.01211433         inf]
[        inf  0.73285413 20.16574242  1.26119082         inf]
[        inf  0.93370946 25.20717803  1.60684885         inf]
[        inf  1.15641785 31.13181845  1.99011443         inf]
[        inf  1.23638288 32.84890759  2.12772867         inf]
[        inf  1.38183969 36.84126019  2.37804969         inf]

Ri
[[ 0.         -0.22150194  0.          0.          0.        ]
 [ 0.          0.         -0.29858508  0.          0.        ]
 [ 0.          0.          0.          0.29858508  0.        ]
 [ 0.          0.          0.          0.          0.22150194]
 [ 0.          0.          0.          0.          0.        ]]
[[ 0.         -0.22456211  0.          0.          0.        ]
 [ 0.          0.         -0.2972335   0.          0.        ]
 [ 0.          0.          0.          0.2972335   0.        ]
 [ 0.          0.          0.          0.          0.22456211]
 [ 0.          0.          0.          0.          0.        ]]
[[ 0.         -0.23359907  0.          0.          0.        ]
 [ 0.          0.         -0.28851412  0.          0.        ]
 [ 0.          0.          0.          0.28851412  0.        ]
 [ 0.          0.          0.          0.          0.23359907]
 [ 0.          0.          0.          0.          0.        ]]
[[ 0.         -0.23656448  0.          0.          0.        ]
 [ 0.          0.         -0.28722302  0.          0.        ]
 [ 0.          0.          0.          0.28722302  0.        ]
 [ 0.          0.          0.          0.          0.23656448]
 [ 0.          0.          0.          0.          0.        ]]
[[ 0.         -0.23950689  0.          0.          0.        ]
 [ 0.          0.         -0.28116768  0.          0.        ]
 [ 0.          0.          0.          0.28116768  0.        ]
 [ 0.          0.          0.          0.          0.23950689]
 [ 0.          0.          0.          0.          0.        ]]
[[ 0.         -0.23950689  0.          0.          0.        ]
 [ 0.          0.         -0.28276671  0.          0.        ]
 [ 0.          0.          0.          0.28276671  0.        ]
 [ 0.          0.          0.          0.          0.23950689]
 [ 0.          0.          0.          0.          0.        ]]
 
0.22150193517531785 0.2985850843731809
0.22456210755125822 0.29723349994112047
0.23359906762293864 0.2885141169419409
0.23656447616946638 0.2872230174589835
0.2395068881899639 0.2811676766529977
0.2395068881899639 0.2827667100113409

0.013677220648274348 0.025321723752706075
0.017043119211226317 0.031459816571280115
0.021714173597089076 0.03932477071410015
0.026893438266979167 0.0485675794867236
0.02875309017444406 0.051246345690000396
0.03213580666555513 0.05747466488983867

"""


# TODO structured output of a,b and f,g coeffs. Done; lgtm

# Thanks TMM package !
def sample_coefficients(pol, n, th_0, freqs):
    a, b = np.zeros_like(freqs, dtype=float), np.zeros_like(freqs, dtype=float)
    f, g = np.zeros_like(freqs, dtype=float), np.zeros_like(freqs, dtype=float)

    lambda_vacs = (c0 / freqs) * 10 ** -6
    for f_idx, lambda_vac in enumerate(lambda_vacs):
        n_list = array(n[f_idx])

        num_layers = n_list.size

        th_list = list_snell(n_list, th_0)

        kz_list = 2 * np.pi * n_list * cos(th_list) / lambda_vac
        f[f_idx], g[f_idx] = kz_list.real[1], kz_list.real[2]

        for i in range(num_layers - 1):
            fresnel_r = interface_r(pol, n_list[i], n_list[i + 1], th_list[i], th_list[i + 1])
            if i == 0:
                b[f_idx] = np.abs(fresnel_r)
            if i == 1:
                a[f_idx] = np.abs(fresnel_r)

    return a, b, f, g


def combine_module_coefficients():
    coeffs = default_coeffs()
    a, b = coeffs[0], coeffs[1]
    """
    self.c0 = self.two * self.a * (self.b * self.b - self.one)
    self.c1 = self.two * self.b
    self.c2 = self.two * self.a * (self.one + self.b * self.b)
    self.c3 = self.two * self.a * self.a * self.b
    self.c4 = self.a * self.a
    self.c5 = self.b * self.b - self.one
    self.c6 = self.b * self.b + self.one
    self.c7 = self.four * self.a * self.b
    """
    c = np.zeros((8, len(selected_freqs)))
    c[0] = 2 * a * (b * b - 1)
    c[1] = 2 * b
    c[2] = 2 * a * (1 + b * b)
    c[3] = 2 * a * a * b
    c[4] = a * a
    c[5] = b * b - 1
    c[6] = b * b + 1
    c[7] = 4 * a * b

    return c


def _verilog_code():
    print("\nVerilog assign f, g: ")
    w = pd + p
    coeffs = default_coeffs()
    fs, gs = coeffs[2], coeffs[3]

    indent = "			"

    for i, f_ in enumerate(fs):
        f_ = numfi(f_ * scaling)
        bin_s = numfi_(i + 1, w=cntr_w, f=0).bin[0]
        if i == 0:
            print(f"assign f = (cntr == 4'b{bin_s}) ? {w}'b{f_.bin[0]}: // {f_} ({f_.w} / {f_.f})")
        else:
            print(indent + f"(cntr == 4'b{bin_s}) ? {w}'b{f_.bin[0]}: // {f_} ({f_.w} / {f_.f})")
    print(indent + "{(4+p){1'b0}};\n")

    for i, g_ in enumerate(gs):
        g_ = numfi(g_ * scaling)
        bin_s = numfi_(i + 1, w=cntr_w, f=0).bin[0]
        if i == 0:
            print(f"assign g = (cntr == 4'b{bin_s}) ? {w}'b{g_.bin[0]}: // {g_} ({g_.w} / {g_.f})")
        else:
            print(indent + f"(cntr == 4'b{bin_s}) ? {w}'b{g_.bin[0]}: // {g_} ({g_.w} / {g_.f})")
    print(indent + "{(4+p){1'b0}};\n")

    c = combine_module_coefficients()
    print("Verilog assign c array: ")
    w = 3 + p

    for c_idx in range(8):
        for i, c_ in enumerate(c[c_idx]):
            c_ = numfi_(c_, s=1, w=3 + p, f=p, rounding="floor")
            bin_s = numfi_(i + pipe_delay, w=cntr_w, f=0).bin[0]
            if i == 0:
                print(f"assign c[{c_idx}] = (cntr == 4'b{bin_s}) ? {w}'b{c_.bin[0]}: // {c_} ({c_.w} / {c_.f})")
            else:
                print(indent + f"(cntr == 4'b{bin_s}) ? {w}'b{c_.bin[0]}: // {c_} ({c_.w} / {c_.f})")
        print(indent + "{(4+p){1'b0}};\n")


def default_coeffs():
    from meas_eval.cw.refractive_index_fit import n

    angle_in = 8 * pi / 180
    freq_idx = [np.argmin(np.abs(f - all_freqs)) for f in selected_freqs]
    one = np.ones_like(selected_freqs)

    n0, n1, n2 = np.transpose(n[freq_idx, 1:4].real)

    # n0 = array([1.513, 1.515, 1.520, 1.521, 1.522, 1.524], dtype=float)
    # n1 = array([2.782, 2.782, 2.784, 2.785, 2.786, 2.787], dtype=float)
    # n2 = array([1.513, 1.515, 1.520, 1.521, 1.522, 1.524], dtype=float)

    n = array([one, n0, n1, n2, one]).T

    return sample_coefficients("s", n, angle_in, selected_freqs)


if __name__ == '__main__':
    np.set_printoptions(floatmode="fixed")

    #### settings #####
    cntr_w = 5
    pd, p = 4, 11
    scaling = 2 ** 3
    pipe_delay = 5
    numfi = partial(numfi_, s=1, w=pd + p, f=p, fixed=True, rounding="floor")

    coeffs = default_coeffs()

    from meas_eval.cw.refractive_index_fit import n
    print(f"Frequencies (THz):\n{selected_freqs}\n")
    freq_idx = [np.argmin(np.abs(f - all_freqs)) for f in selected_freqs]
    print("Refractive index:\n", n[freq_idx, 1:4], "\n")

    #### output #####

    a_s, b_s = str(coeffs[0]).replace(" ", ", "), str(coeffs[1]).replace(" ", ", ")
    f_s, g_s = str(coeffs[2]).replace(" ", ", "), str(coeffs[3]).replace(" ", ", ")

    print(f"a: {a_s}\nb: {b_s}\nf: {f_s}\ng: {g_s}")

    _verilog_code()
