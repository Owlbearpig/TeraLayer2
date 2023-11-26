import tmm
from numpy import array, cos, conj, abs, sqrt
import numpy as np
from scipy.constants import c
from consts import um_to_m, GHz, selected_freqs, n0, n1, n2, f_offset, thea
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
from scipy.optimize import shgo
from scipy.optimize import show_options
from tmm import list_snell, interface_r
from RTL_sim.twos_compl_OF_v2 import real_data_cw
from scipy.special import huber


# show_options("minimize", "Nelder-Mead")


def minimize(*args, **kwargs):
    options_ = {"adaptive": True, "fatol": 0}
    if "x0" in kwargs.keys():
        x0 = kwargs["x0"]
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

    opt_res_ = minimize_(*args, **kwargs, method="Nelder-Mead", options=options_)
    # opt_res_ = minimize_(*args, **kwargs, options=options_)

    return opt_res_


def whitenoise(s=0.05):
    return np.random.uniform(1 - s, 1, size=len(selected_freqs))


randint = np.random.randint
# np.random.seed(37)
noise_scale = 0.0
amp_noise = whitenoise(noise_scale)
phi_noise = whitenoise(noise_scale)

sam_idx = randint(0, 100)
sam_idx = 56

num_layers = 5  # first and last layers are air
pol = "s"
freqs = selected_freqs.copy()
lam = c0 * 1e-6 / freqs
print(f"Frequencies: {freqs} THz,\nwavelengths {np.round(lam, 3)} um")
print(f"Refractive indices: n0={n0},\nn1={n1},\nn2={n2}")

d1, d2, d3 = np.arange(1, 500, 1), np.arange(300, 800, 1), np.arange(1, 500, 1)
n_list = array([np.ones_like(freqs), n0, n1, n2, np.ones_like(freqs)], dtype=float).T

d_truth = [np.inf, 45.77, 660.0, 72.6, np.inf]
d_truth = [np.inf, randint(d1[0], d1[-1]), randint(d2[0], d2[-1]), randint(d3[0], d3[-1]), np.inf]
# d_truth = [np.inf, 270, 469, 327, np.inf]
# d_truth = [np.inf, 29, 568, 111, np.inf]
print(sam_idx, d_truth)

r_fn = np.zeros((len(freqs), num_layers, num_layers), dtype=complex)
kz_list, th_list = np.zeros((2, len(freqs), num_layers), dtype=complex)
r_exp_mod = np.zeros(len(freqs), dtype=complex)
for freq_idx_ in range(freqs.size):
    th_list[freq_idx_] = list_snell(n_list[freq_idx_], thea).T
    kz_list[freq_idx_, :] = 2 * np.pi * n_list[freq_idx_] * cos(th_list[freq_idx_]) / lam[freq_idx_]
    for i in range(num_layers - 1):
        r_fn[freq_idx_, i, i + 1] = interface_r(pol, n_list[freq_idx_, i], n_list[freq_idx_, i + 1],
                                                th_list[freq_idx_, i], th_list[freq_idx_, i + 1])

    r_exp_mod[freq_idx_] = -coh_tmm_slim(pol, n_list[freq_idx_], d_truth, thea, lam[freq_idx_])

r_exp = real_data_cw(sam_idx)  # use real data
r_exp = r_exp_mod  # use model data
print("r_exp: ", r_exp)
r = -np.abs(r_exp) * amp_noise
u = np.exp(1j * np.angle(r_exp) * phi_noise)

c0 = r * r_fn[:, 0, 1] * r_fn[:, 3, 4] * u - r_fn[:, 3, 4]
c1 = r * r_fn[:, 1, 2] * r_fn[:, 3, 4] * u - r_fn[:, 0, 1] * r_fn[:, 1, 2] * r_fn[:, 3, 4]
c2 = (r * r_fn[:, 0, 1] * r_fn[:, 1, 2] * r_fn[:, 2, 3] * u - r_fn[:, 1, 2] * r_fn[:, 2, 3]) * r_fn[:, 3, 4]
c3 = r * r_fn[:, 0, 1] * r_fn[:, 2, 3] * u - r_fn[:, 2, 3]
c4 = r * r_fn[:, 2, 3] * r_fn[:, 3, 4] * u - r_fn[:, 0, 1] * r_fn[:, 2, 3] * r_fn[:, 3, 4]
c5 = r * r_fn[:, 1, 2] * r_fn[:, 2, 3] * u - r_fn[:, 0, 1] * r_fn[:, 1, 2] * r_fn[:, 2, 3]
c6 = r * r_fn[:, 0, 1] * r_fn[:, 1, 2] * u - r_fn[:, 1, 2]
c7 = r * u - r_fn[:, 0, 1]

a0 = c0 * conj(c4) - c3 * conj(c7)
a1 = c1 * conj(c4) - c3 * conj(c6) + c0 * conj(c2) - c5 * conj(c7)
a2 = c1 * conj(c2) - c5 * conj(c6)
a3 = c0 * conj(c1) + c2 * conj(c4) - c3 * conj(c5) - c6 * conj(c7)
a4 = c2 * conj(c1) - c6 * conj(c5)
a5 = abs(c0) + abs(c1) + abs(c2) + abs(c4) - abs(c3) - abs(c5) - abs(c6) - abs(c7)
a6 = c4 * conj(c1) + c2 * conj(c0) - c6 * conj(c3) - c7 * conj(c5)
a7 = c1 * conj(c0) + c4 * conj(c2) - c5 * conj(c3) - c7 * conj(c6)
a8 = c4 * conj(c0) - c7 * conj(c3)


def expr1_(d1_, d2_, freq_idx_=0):
    phi0 = d1_ * kz_list[freq_idx_, 1]
    x_ = np.exp(1j * 2 * phi0)
    phi1 = d2_ * kz_list[freq_idx_, 2]
    y_ = np.exp(1j * 2 * phi1)

    num = c0[freq_idx_] + c1[freq_idx_] * x_ + c2[freq_idx_] * y_ + c4[freq_idx_] * x_ * y_
    den = c3[freq_idx_] + c5[freq_idx_] * x_ + c6[freq_idx_] * y_ + c7[freq_idx_] * x_ * y_

    s = np.abs(num / den)

    return (1 - s)**2


def expr2_(d1_, d3_, freq_idx_=0):
    phi0 = d1_ * kz_list[freq_idx_, 1]
    x_ = np.exp(1j * 2 * phi0)
    phi2 = d3_ * kz_list[freq_idx_, 3]
    z_ = np.exp(1j * 2 * phi2)

    num = c0[freq_idx_] + c1[freq_idx_] * x_ + c3[freq_idx_] * z_ + c5[freq_idx_] * x_ * z_
    den = c2[freq_idx_] + c4[freq_idx_] * x_ + c6[freq_idx_] * z_ + c7[freq_idx_] * x_ * z_

    s = np.abs(num / den)

    return (1 - s)**2


def fun10(x, freq_idx_=0):
    return expr1_(*x, freq_idx_)


def fun11(x):
    return (expr1_(*x, 0) + expr1_(*x, 1) + expr1_(*x, 2) + expr1_(*x, 3) +
            expr1_(*x, 4) + expr1_(*x, 5))


def fun12(x):
    return expr1_(*x, 0) + expr1_(*x, 1)


def fun20(x, freq_idx_=0):
    return expr2_(*x, freq_idx_)


def fun21(x):
    return (expr2_(*x, 0) + expr2_(*x, 1) + expr2_(*x, 2) + expr2_(*x, 3) +
            expr2_(*x, 4) + expr2_(*x, 5))


def fun22(x):
    return expr2_(*x, 0) + expr2_(*x, 1)


def fun1222(x):
    return fun12(x) + fun22(x)


def fun1020(x):
    return fun10(x) + fun20(x)


def fun1121(x):
    return fun11(x) + fun21(x)


def pick_points(points):
    points_sorted = sorted(points, key=lambda x: x[1])
    picked_points = [points_sorted[0]]

    cnt = 0
    for point in points_sorted:
        for already_picked in picked_points:
            if cnt > 5:
                break

            cond_1 = any([abs(already_picked[0][k] - point[0][k]) > 50 for k in range(2)])
            if cond_1:
                cnt += 1
                picked_points.append(point)

    return picked_points


# TODO check total nfev. How do we choose the best starting point?
fun12_freq_idx_0_grid = [(x0_, fun11(x0_)) for x0_ in product(range(d1[0], d1[-1], 10), range(d2[0], d2[-1], 10))]
fun22_freq_idx_0_grid = [(x0_, fun22(x0_)) for x0_ in product(range(d1[0], d1[-1], 50), range(d3[0], d3[-1], 50))]
fun1222_freq_idx_0_grid = [(x0_, fun1222(x0_)) for x0_ in product(range(d1[0], d1[-1], 50), range(d3[0], d3[-1], 50))]
# freq_idx_0_grid = [minimize(fun2, x0=x0_) for x0_ in [(250, i) for i in range(d2[0], d2[-1], 50)]]
print(sorted(fun12_freq_idx_0_grid, key=lambda x: x[1])[:10])
fun12_freq_idx_0_opt_res_sorted = pick_points(fun12_freq_idx_0_grid)
print(fun12_freq_idx_0_opt_res_sorted)
# fun12_freq_idx_0_opt_res_sorted = pick_points(fun12_freq_idx_0_grid)
# fun22_freq_idx_0_opt_res_sorted = pick_points(fun22_freq_idx_0_grid)
# fun1222_freq_idx_0_opt_res_sorted = pick_points(fun22_freq_idx_0_grid)
print("fun12_freq_idx_0_opt_res_sorted:", fun12_freq_idx_0_opt_res_sorted)
# print("fun22_freq_idx_0_opt_res_sorted:", fun22_freq_idx_0_opt_res_sorted[0:3])
# print("fun1222_freq_idx_0_opt_res_sorted:", fun1222_freq_idx_0_opt_res_sorted[0:3])
# print([(x["x"], x["fun"]) for x in freq_idx_0_opt_res_sorted])

tot_nfev = len(fun12_freq_idx_0_grid)
opt_results = []
"""
for x0 in fun12_freq_idx_0_opt_res_sorted:
    # x0 = [int(i) for i in opt_res_["x"]]
    x0 = x0[0]
    d1_x0_0, d1_x0_1 = max(x0[0] - 50, d1[0]), min(x0[0] + 50, d1[-1])
    d2_x0_0, d2_x0_1 = max(x0[1] - 50, d2[0]), min(x0[1] + 50, d2[-1])
    x0s = [(x0_, fun11(x0_)) for x0_ in product(range(d1_x0_0, d1_x0_1, 15), range(d2_x0_0, d2_x0_1, 15))]
    tot_nfev += len(x0s)
    x0s_sorted = sorted(x0s, key=lambda x: x[1])

    for x0_ in x0s_sorted[0:3]:
        opt_res = minimize(fun11, x0=x0_[0])
        tot_nfev += opt_res["nfev"]
        print(opt_res["x"], opt_res["fun"], opt_res["nfev"], tot_nfev)
        opt_results.append(opt_res)
"""
opt_results.append(minimize(fun11, x0=fun12_freq_idx_0_opt_res_sorted[0][0]))

sorted_opt_results = sorted(opt_results, key=lambda x: x["fun"])
best_opt_res = sorted_opt_results[0]
print(best_opt_res)
print(d_truth)
print("total nfev: ", tot_nfev + best_opt_res["nfev"] + len(d3))

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

for freq_idx in range(3):
    X, Z = np.meshgrid(d1, d3)
    vals = fun20([X, Z], freq_idx)
    # Z = np.log10(Z)

    plt.figure()
    plt.title(f"Difference idx: {freq_idx} ({freqs[freq_idx]} THz) fun20")
    plt.imshow(vals,
               extent=[d1[0], d1[-1], d3[0], d3[-1]], origin="lower",
               # interpolation='bilinear',
               # cmap="plasma",
               vmin=np.min(vals), vmax=np.mean(vals),
               )
    plt.xlabel("$d_1$")
    plt.ylabel("$d_3$")

"""
plt.figure()
X, Z = np.meshgrid(d1, d3)
vals = fun21([X, Z])
plt.title(f"Summed differences all idx fun21")
plt.imshow(vals,
           extent=[d1[0], d1[-1], d3[0], d3[-1]],
           origin="lower",
           # interpolation='bilinear',
           # cmap="plasma",
           vmin=np.min(vals), vmax=np.mean(vals),
           )
plt.xlabel("$d_1$")
plt.ylabel("$d_3$")

plt.figure()
X, Z = np.meshgrid(d1, d3)
vals = fun22([X, Z])
plt.title(f"Fun22")
plt.imshow(vals,
           extent=[d1[0], d1[-1], d3[0], d3[-1]],
           origin="lower",
           # interpolation='bilinear',
           # cmap="plasma",
           vmin=np.min(vals), vmax=np.mean(vals),
           )
plt.xlabel("$d_1$")
plt.ylabel("$d_3$")
"""

plt.figure()
plt.title(f"fun10 f0")
y = fun10([d1, d_truth[2] * 0.9])
plt.plot(d1, y)
plt.xlabel("$d_1$")
plt.ylabel("fun10")

for freq_idx in range(3):
    X, Y = np.meshgrid(d1, d2)
    vals = fun10([X, Y], freq_idx)
    # Z = np.log10(Z)

    plt.figure()
    plt.title(f"Difference idx: {freq_idx} ({freqs[freq_idx]} THz) fun10")
    plt.imshow(vals,
               extent=[d1[0], d1[-1], d2[0], d2[-1]], origin="lower",
               # interpolation='bilinear',
               # cmap="plasma",
               vmin=0, vmax=1,
               )
    plt.xlabel("$d_1$")
    plt.ylabel("$d_2$")

plt.figure()
X, Y = np.meshgrid(d1, d2)
vals = fun11([X, Y])
plt.title(f"fun11 (f0 +..+ f5)")
plt.imshow(vals,
           extent=[d1[0], d1[-1], d2[0], d2[-1]],
           origin="lower",
           # interpolation='bilinear',
           # cmap="plasma",
           vmin=np.min(vals), vmax=np.mean(vals),
           )
plt.xlabel("$d_1$")
plt.ylabel("$d_2$")

plt.figure()
X, Y = np.meshgrid(d1, d2)
vals = fun12([X, Y])
plt.title(f"Fun12 (f0 + f1)")
plt.imshow(vals,
           extent=[d1[0], d1[-1], d2[0], d2[-1]],
           origin="lower",
           # interpolation='bilinear',
           # cmap="plasma",
           vmin=np.min(vals), vmax=np.mean(vals),
           )
plt.xlabel("$d_1$")
plt.ylabel("$d_2$")
"""
plt.figure()
X, Z = np.meshgrid(d1, d3)
vals = fun1121([X, Z])
plt.title(f"fun1121")
plt.imshow(vals,
           extent=[d1[0], d1[-1], d3[0], d3[-1]],
           origin="lower",
           # interpolation='bilinear',
           # cmap="plasma",
           vmin=np.min(vals), vmax=np.mean(vals),
           )
plt.xlabel("$d_1$")
plt.ylabel("$d_3$ or $d_2$ I don't know")
"""

plt.figure()
plt.title("(full model - measurement)$^2$, wrt $d_3$")
for opt_res_ in sorted_opt_results:
    r_exp_mod = np.zeros_like(freqs, dtype=complex)
    y_vals = []
    for d3_ in d3:
        for freq_idx_ in range(len(freqs)):
            d = array([np.inf, *opt_res_["x"], d3_, np.inf], dtype=float)
            r_exp_mod[freq_idx_] = -coh_tmm_slim(pol, n_list[freq_idx_], d, thea, lam[freq_idx_])
        err = np.sum((r_exp_mod.real - r_exp.real) ** 2 + (r_exp_mod.imag - r_exp.imag) ** 2)
        y_vals.append(err)

    plt.plot(d3, y_vals)
    min_point = (d3[np.argmin(y_vals)], np.min(y_vals))
    plt.annotate(f"{min_point[0]}, {min_point[1]}", xy=(min_point[0], min_point[1]), xytext=(-20, 20),
                 textcoords='offset points', ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                 color='red'))
plt.xlabel("$d_3$")
plt.ylabel("Loss")

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
