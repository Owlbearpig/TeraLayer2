import tmm
from numpy import array
import numpy as np
from scipy.constants import c
from consts import um_to_m, GHz, selected_freqs, n0, n1, n2
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
from scipy.constants import c as c0
from tmm_package import coh_tmm_slim
from scipy.fftpack import fft, rfft, rfftfreq
from functools import partial
from helpers import multi_root

from scipy.optimize import minimize as minimize_


def minimize(*args, **kwargs):
    opt_res_ = minimize_(*args, **kwargs, method="Nelder-Mead")

    return opt_res_["x"]


freqs = selected_freqs.copy()
lam = c0 * 1e-6 / freqs
print(f"Frequencies: {freqs} THz,\nwavelengths {np.round(lam, 3)} um")
print(f"Refractive indices: n0={n0},\nn1={n1},\nn2={n2}")
d_truth = [np.inf, 60, 450, 240, np.inf]

r0, r1, r2, r3 = (1 - n0) / (1 + n0), (n0 - n1) / (n0 + n1), (n1 - n2) / (n1 + n2), (n2 - 1) / (n2 + 1)

d1, d2, d3 = np.arange(1, 700, 1), np.arange(1, 500, 1), np.arange(1, 500, 1)

r_exp = np.zeros(len(freqs), dtype=complex)
for freq_idx_ in range(freqs.size):
    r_exp[freq_idx_] = coh_tmm_slim("s", [1, n0[freq_idx_], n1[freq_idx_], n2[freq_idx_], 1],
                                    d_truth, 0, lam[freq_idx_])

r, u = np.abs(r_exp), np.exp(1j * np.angle(r_exp))

c0 = r * r0 * r3 * u - r3
c1 = r * r1 * r3 * u - r0 * r1 * r3
c2 = r * r0 * r1 * r2 * r3 * u - r1 * r2 * r3
c3 = r * r0 * r2 * u - r2
c4 = r * r2 * r3 * u - r0 * r2 * r3
c5 = r * r1 * r2 * u - r0 * r1 * r2
c6 = r * r0 * r1 * u - r1
c7 = r * u - r0


def expr1(x, freq_idx_=0):
    d1_, d2_ = x
    phi0 = 2 * np.pi * d1_ * n0[freq_idx_] / lam[freq_idx_]
    x_ = np.exp(1j * 2 * phi0)
    phi1 = 2 * np.pi * d2_ * n1[freq_idx_] / lam[freq_idx_]
    y_ = np.exp(1j * 2 * phi1)

    num = c0[freq_idx_] + c1[freq_idx_] * x_ + c2[freq_idx_] * y_ + c4[freq_idx_] * x_ * y_
    den = c3[freq_idx_] + c5[freq_idx_] * x_ + c6[freq_idx_] * y_ + c7[freq_idx_] * x_ * y_
    s = np.abs(num / den)

    return (1 - s) ** 2


expr1_ = partial(expr1, freq_idx_=0)
# opt_res1 = minimize(expr1_, x0=[50, 50])
opt_res1 = minimize_(expr1, x0=np.array([50, 50]))
print(opt_res1)

for freq_idx in range(6):
    X, Y = np.meshgrid(d1, d2)
    Z = expr1([X, Y], freq_idx)
    # Z = np.log10(Z)

    print(expr1(d_truth[1:3], freq_idx))

    plt.figure()
    plt.title(f"Freq. idx: {freq_idx}")
    plt.imshow(Z, extent=[d1[0], d1[-1], d2[0], d2[-1]], origin="lower",
               # interpolation='bilinear',
               # cmap="plasma",
               vmin=0, vmax=0.5
               )
    plt.xlabel("$d_1$")
    plt.ylabel("$d_2$")

plt.show()
