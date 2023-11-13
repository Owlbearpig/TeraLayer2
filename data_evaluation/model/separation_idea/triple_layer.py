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
d_truth = [np.inf, 175, 350, 40, np.inf]

r0, r1, r2, r3 = (1 - n0) / (1 + n0), (n0 - n1) / (n0 + n1), (n1 - n2) / (n1 + n2), (n2 - 1) / (n2 + 1)

d1, d2, d3 = np.arange(1, 700, 1), np.arange(1, 500, 1), np.arange(1, 500, 1)

r_exp = np.zeros(len(freqs), dtype=complex)
for freq_idx_ in range(freqs.size):
    r_exp[freq_idx_] = coh_tmm_slim("s", [1, n0[freq_idx_], n1[freq_idx_], n2[freq_idx_], 1],
                                    d_truth, 0, lam[freq_idx_])

r, u = np.abs(r_exp), np.exp(1j * np.angle(r_exp))

c0 = r3 - r * r0 * r3 * u
c1 = r0 * r1 * r3 - r * r1 * r3 * u
c2 = r1 * r2 * r3 - r * r0 * r1 * r3 * u
c3 = r2 - r * r0 * r2 * u
c4 = r0 * r2 * r3 - r * r2 * r3 * u
c5 = r0 * r1 * r2 - r * r1 * r2 * u
c6 = r1 - r * r0 * r1 * u
c7 = r0 - r * u


def expr1(d1_, d2_, freq_idx_=0):
    phi0 = 2 * np.pi * d1_ * n0[freq_idx_] / lam[freq_idx_]
    x_ = np.exp(1j * 2 * phi0)
    phi1 = 2 * np.pi * d2_ * n1[freq_idx_] / lam[freq_idx_]
    y_ = np.exp(1j * 2 * phi1)

    num = c0[freq_idx_] + c1[freq_idx_] * x_ + c2[freq_idx_] * y_ + c4[freq_idx_] * x_ * y_
    den = c3[freq_idx_] + c5[freq_idx_] * x_ + c6[freq_idx_] * y_ + c7[freq_idx_] * x_ * y_
    s = np.abs(num / den)

    return (1 - s) ** 2


freq_idx = 0
X, Y = np.meshgrid(d1, d2)
Z = expr1(X, Y, freq_idx)

print(expr1(175.0, 350.0))

fig0, ax0 = plt.subplots(subplot_kw={"projection": "3d"})
ax0.set_title(f"Expression 1 (Fidx: {freq_idx})")
surf0 = ax0.plot_surface(X, Y, Z, cmap=cm.viridis, antialiased=True)
ax0.set_xlabel("$d_1$")
ax0.set_ylabel("$d_2$")
ax0.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax0.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig0.colorbar(surf0, shrink=1, aspect=5)

plt.show()
