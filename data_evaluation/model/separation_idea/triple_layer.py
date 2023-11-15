import tmm
from numpy import array
import numpy as np
from scipy.constants import c
from consts import um_to_m, GHz, selected_freqs, n0, n1, n2, f_offset
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
from scipy.constants import c as c0
from tmm_package import coh_tmm_slim
from scipy.fftpack import fft, rfft, rfftfreq
from functools import partial
from helpers import multi_root
from itertools import product
from scipy.optimize import minimize as minimize_
from scipy.optimize import show_options


# show_options("minimize", "Nelder-Mead")


def minimize(*args, **kwargs):
    options_ = {"adaptive": True}
    if "x0" in kwargs.keys():
        x0 = kwargs["x0"]
        print(x0)
        nonzdelt = 0.025
        zdelt = 0.00025
        N = len(x0)

        sim = np.empty((N + 1, N), dtype=float)
        sim[0] = x0
        for k in range(N):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt) * y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
        options_["initial_simplex"] = sim
        print(sim)

    opt_res_ = minimize_(*args, **kwargs, method="Nelder-Mead", options=options_)

    return opt_res_


def whitenoise(s=0.05):
    return np.random.uniform(1 - s, 1, size=len(selected_freqs))


noise_scale = 0.00
amp_noise = whitenoise(noise_scale)
phi_noise = whitenoise(noise_scale)

# np.random.seed(42)

# selected_freqs = array([0.020, 0.100, 0.210, 0.250, 0.340, 0.480], dtype=float)
# selected_freqs = np.random.uniform(0.1, 1.2, size=6)
# selected_freqs.sort()

freqs = selected_freqs.copy()
lam = c0 * 1e-6 / freqs
print(f"Frequencies: {freqs} THz,\nwavelengths {np.round(lam, 3)} um")
print(f"Refractive indices: n0={n0},\nn1={n1},\nn2={n2}")
d_truth = [np.inf, 45, 640, 65, np.inf]

# TODO add angle dependencies. + Check sign
r0, r1, r2, r3 = (1 - n0) / (1 + n0), (n0 - n1) / (n0 + n1), (n1 - n2) / (n1 + n2), (n2 - 1) / (n2 + 1)

d1, d2, d3 = np.arange(1, 500, 1), np.arange(300, 800, 1), np.arange(1, 500, 1)

r_exp = np.zeros(len(freqs), dtype=complex)
for freq_idx_ in range(freqs.size):
    r_exp[freq_idx_] = -coh_tmm_slim("s", [1, n0[freq_idx_], n1[freq_idx_], n2[freq_idx_], 1],
                                     d_truth, 0, lam[freq_idx_])
print(r_exp)
# r_exp = np.array([0.68861143 + 0.19188424j, 0.40914245 + 0.21957212j, 0.27207933 - 0.37308845j, -0.17356888 - 0.2019538j,0.21104934 - 0.13649972j, 0.03111117 + 0.24119368j], dtype=complex)
print(r_exp)
r = np.abs(r_exp) * amp_noise
u = np.exp(1j * np.angle(r_exp) * phi_noise)

c0 = r * r0 * r3 * u - r3
c1 = r * r1 * r3 * u - r0 * r1 * r3
c2 = r * r0 * r1 * r2 * r3 * u - r1 * r2 * r3
c3 = r * r0 * r2 * u - r2
c4 = r * r2 * r3 * u - r0 * r2 * r3
c5 = r * r1 * r2 * u - r0 * r1 * r2
c6 = r * r0 * r1 * u - r1
c7 = r * u - r0


def expr1_(d1_, d2_, freq_idx_=0):
    phi0 = 2 * np.pi * d1_ * n0[freq_idx_] / lam[freq_idx_]
    x_ = np.exp(1j * 2 * phi0)
    phi1 = 2 * np.pi * d2_ * n1[freq_idx_] / lam[freq_idx_]
    y_ = np.exp(1j * 2 * phi1)

    num = c0[freq_idx_] + c1[freq_idx_] * x_ + c2[freq_idx_] * y_ + c4[freq_idx_] * x_ * y_
    den = c3[freq_idx_] + c5[freq_idx_] * x_ + c6[freq_idx_] * y_ + c7[freq_idx_] * x_ * y_

    s = np.abs(num / den)

    return abs(1 - s)


def fun(x, freq_idx_=0):
    return expr1_(*x, freq_idx_)


def fun1(x):
    return (expr1_(*x, 0) + expr1_(*x, 1) + expr1_(*x, 2) + expr1_(*x, 3) +
            expr1_(*x, 4) + expr1_(*x, 1))


# from scipy.optimize import shgo
# bounds = [(0, 500), (0, 500)]
x0s = [x0_ for x0_ in product(range(d1[0], d1[-1], 20), range(d2[0], d2[-1], 20))]
y0s = [(x0_, fun1(x0_)) for x0_ in x0s]
print(len(y0s))
x0_best = sorted(y0s, key=lambda x: x[1])[0]
print(x0_best)
opt_res = minimize(fun1, x0=x0_best[0])
print(opt_res["nfev"])
tot_nfev = opt_res["nfev"] + len(y0s)
print(opt_res["x"], opt_res["fun"], opt_res["nfev"], tot_nfev)
print(d_truth)

"""
fun0, best_x0 = np.inf, None
for x0 in x0s:
    opt_res1 = minimize_(fun1, x0=x0)
    # opt_res1 = shgo(expr1_, bounds, options={"f_min": 1e-5})
    if opt_res1["fun"] < fun0:
        fun0 = opt_res1["fun"]
        best_x0 = opt_res1["x"]
    tot_nfev += opt_res1["nfev"]
    print(opt_res1["x"], opt_res1["fun"], opt_res1["nfev"])
print(best_x0, fun0, tot_nfev)
"""

for freq_idx in range(6):
    X, Y = np.meshgrid(d1, d2)
    Z = fun([X, Y], freq_idx)
    # Z = np.log10(Z)

    plt.figure()
    plt.title(f"Summed differences idx: {freq_idx}")
    plt.imshow(Z,
               extent=[d1[0], d1[-1], d2[0], d2[-1]], origin="lower",
               # interpolation='bilinear',
               # cmap="plasma",
               vmin=np.min(Z), vmax=np.mean(Z),
               )
    plt.xlabel("$d_1$")
    plt.ylabel("$d_2$")

X, Y = np.meshgrid(d1, d2)
Z = fun1([X, Y])
# Z = np.log10(Z)

plt.figure()
plt.title(f"Summed differences idx: sum")
plt.imshow(Z,
           extent=[d1[0], d1[-1], d2[0], d2[-1]],
           origin="lower",
           # interpolation='bilinear',
           # cmap="plasma",
           vmin=np.min(Z), vmax=np.mean(Z),
           )
plt.xlabel("$d_1$")
plt.ylabel("$d_2$")
"""
from scipy import fftpack, ndimage
plt.figure()
fft2 = fftpack.fft2(Z)
plt.imshow(np.log10(np.abs(fft2)),
    vmin=3, vmax=4.5,
           )
"""
"""
Z = np.gradient(Z)

plt.figure()
plt.title(f"Gradient")
plt.imshow(Z[0], extent=[d1[0], d1[-1], d2[0], d2[-1]], origin="lower",
           # interpolation='bilinear',
           # cmap="plasma",
           vmin=0, vmax=0.001,
           )
plt.xlabel("$d_1$")
plt.ylabel("$d_2$")
"""
plt.show()
