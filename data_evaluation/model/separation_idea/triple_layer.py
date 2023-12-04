import matplotlib as mpl
import tmm
from numpy import array, cos, conj, abs, sqrt, sin
import numpy as np
from scipy.constants import c
from scipy.signal import argrelextrema
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
tot_nfev = 0


def minimize(*args, **kwargs):
    options_ = {"adaptive": True, "fatol": 0}
    if "x0" in kwargs.keys():
        x0 = kwargs["x0"]
        nonzdelt = 0.025 * 20
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

    if "bounds" not in kwargs.keys():
        kwargs["bounds"] = [(0, d1[-1]), (0, d2[-2])]

    opt_res_ = minimize_(*args, **kwargs, method="Nelder-Mead", options=options_)
    # opt_res_ = minimize_(*args, **kwargs, options=options_)

    opt_res_["x"] = opt_res_["x"].astype(int)

    return opt_res_


def whitenoise(s=0.05):
    return np.random.uniform(1 - s, 1, size=len(selected_freqs))


randint = np.random.randint
seed = randint(0, 10000, size=1)
# seed = 3736  # doesn't work with first algo idea # works "xy idea" (barely) should assume a and shift are fixed.
# seed = 3191  # difficult # works "xy idea"
# seed = 924 # works "xy idea"
# seed = 4435  # ellipse seed works "xy idea"
# seed = 7542  # doesnt work # works "xy idea"
# seed = 1234 # works "xy idea"
# seed = 6063 # works "xy idea"
# seed = 5577
# seed = 1768
# seed = 3454
seed = 7217
np.random.seed(seed)
print(seed)

noise_scale = 0.00
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

d1, d2, d3 = np.arange(0, 500, 1), np.arange(0, 500, 1), np.arange(0, 500, 1)
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

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
ax0.set_title("Amplitude")
ax1.set_title("Phase")
ax1.set_xlabel("Wavelength (um)")
ax0.set_ylabel("Amplitude (arb. u.)")
ax1.set_ylabel("Phase (rad)")

ax0.plot(lam, r, label="Truth - Noisy")
ax0.plot(lam, r / amp_noise, label="Truth - 0 noise")

ax1.plot(lam, np.angle(r_exp) * phi_noise, label="Truth - Noisy")
ax1.plot(lam, np.angle(r_exp), label="Truth - 0 noise")

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

    return (1 - s) ** 2


def expr1xy_(d1_, d2_, freq_idx_=0):
    phi0 = d1_ * kz_list[freq_idx_, 1]
    x_ = np.exp(1j * 2 * phi0)
    phi1 = d2_ * kz_list[freq_idx_, 2]
    y_ = np.exp(1j * 2 * phi1)

    num = c0[freq_idx_] + c4[freq_idx_] * x_ * y_
    den = c3[freq_idx_] + c7[freq_idx_] * x_ * y_

    s = np.abs(num / den)

    return (1 - s) ** 2


def expr2_(d1_, d3_, freq_idx_=0):
    phi0 = d1_ * kz_list[freq_idx_, 1]
    x_ = np.exp(1j * 2 * phi0)
    phi2 = d3_ * kz_list[freq_idx_, 3]
    z_ = np.exp(1j * 2 * phi2)

    num = c0[freq_idx_] + c1[freq_idx_] * x_ + c3[freq_idx_] * z_ + c5[freq_idx_] * x_ * z_
    den = c2[freq_idx_] + c4[freq_idx_] * x_ + c6[freq_idx_] * z_ + c7[freq_idx_] * x_ * z_

    s = np.abs(num / den)

    return (1 - s) ** 2


def fun10(x, freq_idx_=0):
    return expr1_(*x, freq_idx_)


def fun11(x):
    return (expr1_(*x, 0) + expr1_(*x, 1) + expr1_(*x, 2) + expr1_(*x, 3) +
            expr1_(*x, 4) + expr1_(*x, 5))


def fun12(x):
    return expr1_(*x, 3) + expr1_(*x, 4) + expr1_(*x, 5)


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


def pick_points(points, cnt=2):
    points_sorted = sorted(points, key=lambda x: x[1])
    picked_points = [points_sorted[0]]

    for point in points_sorted:
        for already_picked in picked_points:
            if len(picked_points) >= cnt:
                break

            cond_1 = any([abs(already_picked[0][k] - point[0][k]) > 50 for k in range(2)])
            if cond_1:
                picked_points.append(point)

    return picked_points


minima = []

for i in range(0, 200, 5):
    y = expr1xy_(*[d1, i], 2)
    ext = argrelextrema(y, np.less)
    minima.append((d1[ext[0][0]], i))
    print(ext)

y = expr1xy_(*[200, d2], 2)
ext = argrelextrema(y, np.less)
shift = 83  # d2[ext[0][1]] - d2[ext[0][0]]

y = expr1xy_(*[0, d2], 2)
ext = argrelextrema(y, np.less)
y0 = d2[ext[0][0]]

print(minima)
plt.figure("minima")
plt.plot([i[0] for i in minima], [i[1] for i in minima])
plt.xlabel("d1")
plt.ylabel("d2")

minima_d1 = np.array([i[0] for i in minima], dtype=int)
minima_d2 = np.array([i[1] for i in minima], dtype=int)
minima_grad = np.gradient(minima_d2, minima_d1)
print(minima_grad)
a = np.mean(minima_grad[:5])
print(a, shift)

diag_lines = []
for i in range(10):
    for d1_ in range(0, 500, 5):
        d2_ = d1_ * a + i * shift + y0
        if 0 < d2_ < d2[-1]:
            diag_lines.append((d1_, d2_))

tot_nfev = 0

large_grid = []
for i in range(-10, 15, 5):
    large_grid.extend([(pt[0], pt[1] - i) for pt in diag_lines])

all_points = sorted([(pt, fun11(pt)) for pt in large_grid], key=lambda x: x[1])
tot_nfev += len(all_points)
print(all_points[:5])
best_point = all_points[0]

final_opt_res = minimize(fun11, x0=best_point[0])

print(final_opt_res)
print(d_truth)
print(tot_nfev + final_opt_res["nfev"])


plt.figure()
plt.title(f"fun10 f0")
y = fun10([d1, 100], 2)
plt.plot(d1, y, label=f"fun10(d, {100})")
plt.xlabel("d")
plt.ylabel("fun10")

for freq_idx in range(6):
    X, Y = np.meshgrid(d1, d2)
    vals = fun10([X, Y], freq_idx)
    # Z = np.log10(Z)

    plt.figure(f"f{freq_idx}")
    plt.title(f"Difference idx: {freq_idx} ({freqs[freq_idx]} THz) fun10")
    plt.imshow(vals,
               extent=[d1[0], d1[-1], d2[0], d2[-1]], origin="lower",
               # interpolation='bilinear',
               # cmap="plasma",
               vmin=0, vmax=np.mean(vals),
               )
    plt.xlabel("$d_1$")
    plt.ylabel("$d_2$")

plt.figure()
X, Y = np.meshgrid(d1, d2)
vals = fun12([X, Y])
plt.title(f"Fun12 (f0 + f1 + f2)")
plt.imshow(vals,
           extent=[d1[0], d1[-1], d2[0], d2[-1]],
           origin="lower",
           # interpolation='bilinear',
           # cmap="plasma",
           vmin=0, vmax=np.mean(vals),
           )
plt.xlabel("$d_1$")
plt.ylabel("$d_2$")

sum_expr1xy_vals = np.zeros_like(vals)
for i in range(len(lam)):
    plt.figure(f"expr1xy_f{i}")
    X, Y = np.meshgrid(d1, d2)
    vals = expr1xy_(*[X, Y], i)
    plt.title(f"expr1xy f{i}")
    plt.imshow(vals,
               extent=[d1[0], d1[-1], d2[0], d2[-1]],
               origin="lower",
               # interpolation='bilinear',
               # cmap="plasma",
               vmin=0, vmax=np.mean(vals),
               )
    plt.xlabel("$d_1$")
    plt.ylabel("$d_2$")
    sum_expr1xy_vals += vals

    plt.figure(f"expr1xy_1D_sliced")
    vals = expr1xy_(100, d2, i)
    plt.title(f"expr1xy f{i}")
    plt.plot(d2, vals, label=f"expr1xy_f{i}(100, d2)")
    plt.xlabel("$d_2$")
    plt.ylabel("value")

plt.figure(f"expr1xy_summed_1D_sliced")
plt.title(f"expr1xy summed")
for d1_ in range(0, 600, 100):
    f_sum_expr1xy = np.sum([expr1xy_(d1_, d2, i) for i in range(6)], axis=0)
    plt.plot(d2, f_sum_expr1xy, label=f"expr1xy({d1_}, d2)_f_sum")
plt.xlabel("$d_2$")
plt.ylabel("value")

print("##############################################################")
sum_expr1xy_1D_0 = np.sum([expr1xy_(0, d2, i) for i in range(6)], axis=0)
sum_expr1xy_1D_500 = np.sum([expr1xy_(500, d2, i) for i in range(6)], axis=0)
a = -0.5429

if np.min(sum_expr1xy_1D_0) < np.min(sum_expr1xy_1D_500):
    print(np.min(sum_expr1xy_1D_0))
    shift = d2[np.argmin(sum_expr1xy_1D_0)]
    b = shift
else:
    print(np.min(sum_expr1xy_1D_500))
    shift = d2[np.argmin(sum_expr1xy_1D_500)]
    b = shift - a*d2[-1]
print(shift, b)

diag_line = []
for d1_ in range(0, d1[-1], 5):
    d2_ = d1_ * a + b
    if 0 < d2_ < d2[-1]:
        diag_line.append((d1_, d2_))

#print(diag_line)
#diag_line = list(zip(np.arange(0, d1[-1], 5), b + a*np.arange(0, d1[-1], 5)))
#print(diag_line)

tot_nfev = 2*len(sum_expr1xy_1D_0)
grid = []
for i in range(-10, 15, 5):
    grid.extend([(pt[0], pt[1] - i) for pt in diag_line])

all_points = sorted([(pt, fun11(pt)) for pt in grid], key=lambda x: x[1])
tot_nfev += len(all_points)
print(all_points[:5])
best_point = all_points[0]

final_opt_res = minimize(fun11, x0=best_point[0])

print(final_opt_res)
print(d_truth)
print(tot_nfev + final_opt_res["nfev"])

plt.figure("expr1xy summed")
plt.title(f"expr1xy summed")
plt.imshow(sum_expr1xy_vals,
           extent=[d1[0], d1[-1], d2[0], d2[-1]],
           origin="lower",
           # interpolation='bilinear',
           # cmap="plasma",
           vmin=0, vmax=np.mean(sum_expr1xy_vals),
           )
plt.xlabel("$d_1$")
plt.ylabel("$d_2$")


def ellipse():
    t = np.linspace(0, 2 * np.pi, 100)
    a, b = 75, 40
    h, k = 140, 210
    A = -15 * np.pi / 180
    x = cos(A) * a * cos(t) - sin(A) * b * sin(t) + h
    y = sin(A) * a * cos(t) + cos(A) * b * sin(t) + k

    return list(zip(x, y))


plt.figure("fun11_f0-f5")
for pt in grid:
    plt.scatter(*pt, s=1, c='black', marker='o')
"""
plt.figure("fun11_f0-f5")
for pt in large_grid:
    plt.scatter(*pt, s=1, c='black', marker='o')

plt.figure("expr1xy")
for pt in large_grid:
    plt.scatter(*pt, s=1, c='black', marker='o')
"""



plt.figure("fun11_f0-f5")
X, Y = np.meshgrid(d1, d2)
vals = fun11([X, Y])
plt.title(f"fun11 (f0 +..+ f5)")
plt.imshow(vals,
           extent=[d1[0], d1[-1], d2[0], d2[-1]],
           origin="lower",
           # interpolation='bilinear',
           # cmap="plasma",
           vmin=0, vmax=np.mean(vals),
           )
plt.xlabel("$d_1$")
plt.ylabel("$d_2$")

"""
for pt in diag_lines:
    plt.scatter(*pt, s=1, c='black', marker='o')
"""



plt.figure()
plt.title("(full model - measurement)$^2$, wrt $d_3$")
r_exp_mod = np.zeros_like(freqs, dtype=complex)
y_vals = []
best_fit = (None, np.inf)
for d3_ in d3:
    for freq_idx_ in range(len(freqs)):
        d = array([np.inf, *final_opt_res["x"], d3_, np.inf], dtype=float)
        r_exp_mod[freq_idx_] = -coh_tmm_slim(pol, n_list[freq_idx_], d, thea, lam[freq_idx_])
    err = np.sum((r_exp_mod.real - r_exp.real) ** 2 + (r_exp_mod.imag - r_exp.imag) ** 2)
    y_vals.append(err)
    if err < best_fit[1]:
        best_fit = (r_exp_mod, err)

plt.plot(d3, y_vals, label="Squared differences")
min_point = (d3[np.argmin(y_vals)], np.min(y_vals))
plt.annotate(f"{min_point[0]}, {min_point[1]}", xy=(min_point[0], min_point[1]), xytext=(-20, 20),
             textcoords='offset points', ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                             color='red'))
plt.xlabel("$d_3$")
plt.ylabel("Loss")

ax0.plot(lam, -np.abs(best_fit[0]), label="Found")
ax1.plot(lam, np.angle(best_fit[0]), label="Found")

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
for fig_num in plt.get_fignums():
    plt.figure(fig_num)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        plt.legend()

    axes = fig.get_axes()
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

plt.show()
