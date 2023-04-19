import matplotlib.pyplot as plt
import numpy as np
from numpy import array, pi
from scipy.constants import c as c0
from meas_eval.cw.load_data import transfer_function
from tmm_package import coh_tmm_slim, coh_tmm_slim_unsafe
from functions import do_ifft, filtering, window, do_fft, zero_pad, shift
from scipy.optimize import shgo
from scipy.optimize import minimize
from matplotlib.widgets import Slider, Button, RangeSlider, TextBox



def filter(data_td, en=True):
    if not en:
        return data_td

    data_td = filtering(data_td, filt_type="hp", wn=0.22, order=1)
    # data_td = filtering(data_td, filt_type="lp", wn=1.76, order=4)
    data_td = filtering(data_td, filt_type="lp", wn=2.10, order=7)

    return data_td

sam_idx = None

if sam_idx is None:
    try:
        t_func_fd = np.load("t_func_mean.npy")
        freqs = t_func_fd[:, 0].real
        amp_meas_mean = np.abs(t_func_fd[:, 1])
        phi_meas_mean = np.angle(t_func_fd[:, 1])
    except FileNotFoundError:
        amp_meas_all, angle_meas_all = [], []
        for sam_idx in range(101):
            # sam_idx = 23

            t_func_fd = transfer_function(sam_idx)
            # t_func_fd = zero_pad(t_func_fd, mult=pad)

            """
            plt.figure("test")
            plt.plot(np.angle(t_func_fd), label="angle before")
            plt.figure("test2")
            plt.plot(np.abs(t_func_fd), label="magn before")
            """

            amp_meas_all.append(np.abs(t_func_fd[:, 1]))
            angle_meas_all.append(np.angle(t_func_fd[:, 1]))

        amp_meas_all = array(amp_meas_all)
        angle_meas_all = array(angle_meas_all)

        amp_meas_ = np.mean(amp_meas_all, axis=0)
        angle_meas_ = np.mean(angle_meas_all, axis=0)

        t_func_fd = transfer_function(0)
        t_func_all_fd = array([t_func_fd[:, 0].real, amp_meas_mean * np.exp(1j * angle_meas_mean)]).T

        np.save(f"t_func_mean.npy", t_func_all_fd)


else:
    t_func_fd = transfer_function(sam_idx)
    angle_meas_ = np.angle(t_func_fd[:, 1])
    amp_meas_ = np.abs(t_func_fd[:, 1])

freqs = t_func_fd[:, 0].real
# amp_meas_ = np.convolve(amp_meas_, np.ones(7)/7, mode='same')
# angle_meas_ = np.convolve(angle_meas_, np.ones(3)/3, mode='same')

t_func_td = do_ifft(t_func_fd, flip=False)
t_func_td = shift(t_func_td, -100)
t_func_td = filter(t_func_td)
t_func_td = shift(t_func_td, 100)
# t_func_td = window(t_func_td, win_start=shift-5, win_width=25, en_plot=False, slope=0.3)
dt = np.mean(np.diff(t_func_td[:, 0]))

# t_func_fd = do_fft(t_func_td)
t_func_td = shift(t_func_td, 100)

"""
plt.figure()
for i in range(101):
    continue
    plt.plot(freqs, angle_meas_all[i])
    plt.plot(freqs, amp_meas_all[i])

plt.plot(freqs, angle_meas_avg, label="average", color="black")
plt.plot(freqs, amp_meas_avg, label="average", color="black")
plt.legend()
"""
"""
plt.figure("test")
plt.plot(np.angle(t_func_fd), label="angle after")
plt.legend()
plt.figure("test2")
plt.plot(np.abs(t_func_fd), label="magn after")
plt.legend()
plt.show()
"""

f0, f1 = 0.150, 1.200
f0_idx, f1_idx = np.argmin(np.abs(freqs - f0)), np.argmin(np.abs(freqs - f1))

np.random.seed(420)

bounds = ((1.50, 1.65), (2.75, 2.85),)
bounds = ((1.58, 1.62), (2.75, 2.85), (1.58, 1.62))
# bounds = ((3.00, 3.10), )
# bounds = ((1.55, 1.65), )

# p_sol = array([np.inf, 46.0, 641.0, 79.0, np.inf])
one = np.ones_like(freqs)

n = array([one, 1.6 * one, 2.80 * one, 1.6 * one, one]).T

k0 = np.linspace(0.000, 0.050, len(one))
k1 = np.linspace(0.000, 0.050, len(one))
k2 = np.linspace(0.000, 0.050, len(one))

n = n - 1j * np.array([np.zeros_like(one), k0, k1, k2, np.zeros_like(one)]).T

"""
fa_idx, fe_idx = np.argmin(np.abs(freqs - 0.050)), np.argmin(np.abs(freqs - 0.5))
n[fa_idx:fe_idx, 2] = np.linspace(2.80, 2.80, fe_idx - fa_idx)

fa_idx, fe_idx = np.argmin(np.abs(freqs - 0.5)), np.argmin(np.abs(freqs - 2.00))
n[fa_idx:fe_idx, 2] = np.linspace(2.80, 2.80, fe_idx - fa_idx)

fa_idx, fe_idx = np.argmin(np.abs(freqs - 0.050)), np.argmin(np.abs(freqs - 1.00))
n[fa_idx:fe_idx, 1] = np.linspace(1.60, 1.60, fe_idx - fa_idx)

fa_idx, fe_idx = np.argmin(np.abs(freqs - 0.050)), np.argmin(np.abs(freqs - 1.00))
n[fa_idx:fe_idx, 3] = np.linspace(1.60, 1.60, fe_idx - fa_idx)

fa_idx, fe_idx = np.argmin(np.abs(freqs - 1.0)), np.argmin(np.abs(freqs - 2.00))
n[fa_idx:fe_idx, 1] = np.linspace(1.60, 1.60, fe_idx - fa_idx)

fa_idx, fe_idx = np.argmin(np.abs(freqs - 1.0)), np.argmin(np.abs(freqs - 2.00))
n[fa_idx:fe_idx, 3] = np.linspace(1.60, 1.60, fe_idx - fa_idx)

k = np.linspace(0, 0.150, len(one))
"""

# n0_truth = np.random.uniform(*bounds[0], len(freqs))
# n1_truth = np.random.uniform(*bounds[1], len(freqs))
# n0_truth = 1.65 * one
# n1_truth = 2.86 * one
# n_truth = array([one, n0_truth, n1_truth, n0_truth, one]).T
# n_truth = array([one, 1.692027738932626*one, 2.884941014415432*one, 1.692027738932626*one, one]).T
# n_truth = array([one, 1.7*one, 2.8*one, 1.7*one, one]).T

for f_idx in range(len(freqs)):
    continue
    lam_vac = 10 ** 6 * c0 / (freqs[f_idx] * 10 ** 12)
    t_func_fd[f_idx, 1] = -1 * coh_tmm_slim("s", n_truth[f_idx], p_sol, 8 * pi / 180, lam_vac)

"""
plt.figure("test")
plt.plot(np.angle(t_func_fd), label="angle before")
plt.figure("test2")
plt.plot(np.abs(t_func_fd), label="magn before")

t_func_td = do_ifft(t_func_fd, flip=False)
t_func_td = filter(t_func_td)
t_func_fd = do_fft(t_func_td)
t_func_td = shift(t_func_td, 100)

plt.figure("test")
plt.plot(np.angle(t_func_fd), label="angle after")
plt.legend()
plt.figure("test2")
plt.plot(np.abs(t_func_fd), label="magn after")
plt.legend()
plt.show()
"""

t_func_td[:, 1] /= np.max(np.abs(t_func_td[:, 1]))


def freq_fit(thicknesses):
    def cost(p, f_idx):
        # p_sol_ = array([np.inf, 46.0, 637.0, 79.0, np.inf])
        # p_sol_ = p_sol.copy()
        # n_list = array([1, p[0], p[1], p[0], 1])
        n_list = array([1, p[0], p[1], p[0], 1])
        # n_list = array([1, p[0], p[1], p[0], 1])
        # n_list = array([1, 1.6, p[0], p[1], 1])
        # n_list = array([1, p[0], 2.8])

        loss_phi, loss_amp = 0, 0
        for i in range(-0, 1):
            lam_vac = 10 ** 6 * c0 / (freqs[f_idx + i] * 10 ** 12)
            mod_fd = -1 * coh_tmm_slim_unsafe("s", n_list, thicknesses, 8 * pi / 180, lam_vac)

            loss_phi += (angle_meas_[f_idx + i] - np.angle(mod_fd)) ** 2
            loss_amp += (amp_meas_[f_idx + i] - np.abs(mod_fd)) ** 2

        """
        lam_vac = 10 ** 6 * c0 / (freqs[f_idx] * 10 ** 12)
        mod_fd = -1 * coh_tmm_slim_unsafe("s", n_list, thicknesses, 8 * pi / 180, lam_vac)

        loss_phi = (angle_meas_avg[f_idx] - np.angle(mod_fd)) ** 2
        loss_amp = (amp_meas_avg[f_idx] - np.abs(mod_fd)) ** 2
        """

        return loss_phi + loss_amp

    """
    #x, y = np.linspace(1.69201, 1.69203, 200), np.linspace(2.88493, 2.88495, 200)
    x, y = np.linspace(1.5, 2, 200), np.linspace(2.5, 3.5, 200)
    x_long, y_long = np.linspace(1.5, 3, 1000), np.linspace(2.5, 3.5, 1000)

    plt.figure()
    
    Frequency 1.4629999999999999 (1697/4500)
    Result: [1.6875,    2.8852679] (3.2237342661786996e-05)
    truth: [1.692027738932626, 2.884941014415432]
    """
    """
    f_idx_ = 1697 - 600
    print(freqs[f_idx_])
    plt.plot(x_long, [cost([x_, 2.81], f_idx_) for x_ in x_long])
    plt.plot(x_long, [cost([x_, 2.8], f_idx_) for x_ in x_long])
    plt.plot(y_long, [cost([1.7, y_], f_idx_) for y_ in y_long])
    plt.plot(y_long, [cost([1.6, y_], f_idx_) for y_ in y_long])
    plt.show()
    plt.figure()
    grid = np.zeros((len(x), len(y)))
    for i, p0 in enumerate(x):
        for j, p1 in enumerate(y):
            grid[i, j] = (cost([p0, p1], 1697))
    plt.imshow(grid, extent=[x[0], x[-1], y[0], y[-1]], origin="lower")
    #plt.show()
    print(cost([1.692,    2.884941014415432], 1697))
    """
    shgo_en = False
    # n1_res, n2_res = [], []
    for f_idx, freq in enumerate(freqs):
        # if (freq < f0) or (freq > f1) or (f_idx % 10 != 0):
        if (freq < f0) or (freq > f1):
            continue
        print(f"Frequency {freq} ({f_idx}/{len(freqs)})")
        if shgo_en:
            iters = 3
            res = shgo(cost, args=(f_idx,), bounds=bounds, iters=iters, n=2 ** 8)
            while res.fun > 1e-10:
                iters += 1
                if iters == 5:
                    break
                res = shgo(cost, args=(f_idx,), bounds=bounds, iters=iters, n=2 ** 12)
        else:
            x0 = n[f_idx, 1:4]
            res = minimize(cost, x0, args=(f_idx,), bounds=bounds, tol=0.01)

        print(f"Result: {res.x} ({res.fun})")
        # print(f"truth: [{n0_truth[f_idx]}, {n1_truth[f_idx]}]")
        # n[f_idx, 1], n[f_idx, 2], n[f_idx, 3] = res.x
        n[f_idx, 1], n[f_idx, 2], n[f_idx, 3] = res.x[0], res.x[1], res.x[2]
        # n1_res.append(res.x[0]), n2_res.append(res.x[1])

        # n[f_idx, 2] = res.x[0]
        """
        if res.fun < 1e-3:
            #n[f_idx, 1], n[f_idx, 2], n[f_idx, 3] = res.x[0], res.x[0], res.x[0],
            n[f_idx, 2], n[f_idx, 3] = res.x[0], res.x[1]
        else:
            continue
        """
        # n[f_idx, 1], n[f_idx, 2], n[f_idx, 3] = 0, 0, 0
        # n[f_idx, 1], n[f_idx, 2], n[f_idx, 3] = 1.5, res.x[0], 1.5,
    """
    plt.figure("Refractive index fitted")
    plt.plot(freqs, n[:, 1], label="n0")
    plt.plot(freqs, n[:, 2], label="n1")
    plt.plot(freqs, n[:, 3], label="n2")
    plt.axvline(x=f0, color="b")
    plt.axvline(x=f1, color="b", label="Fit range")
    plt.legend()
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive index")
    """

    """

    # n1_res = np.concatenate((n1_res, n1_res[-1]*np.ones(1000)))
    x_axis = np.fft.rfftfreq(n=len(n1_res))
    plt.figure("test")
    plt.title(f"{thicknesses}")
    plt.plot(n1_res)
    plt.plot(x_axis[1:], np.abs(np.fft.rfft(n1_res-np.mean(n1_res)))[1:])
    plt.show()
    """
    # n1_res = np.convolve(n1_res, np.ones(5) / 5, mode='valid')
    # gof = np.mean(np.abs(np.fft.rfft(n1_res))[1:])
    # gof = np.max(np.abs(np.fft.rfft(n1_res - np.mean(n1_res)))[1:])

    # gof = np.std(n1_res) + np.std(n2_res)

    return n


p0 = array([np.inf, 44.62, 660, 75.9, np.inf])  # truth: array([np.inf, 46.0, 641.0, 79.0, np.inf])
# p0 = array([np.inf, 0.0, 0.0, 0.0, np.inf])

p = p0 + array([0, 0, 0, 0, 0])
# n = freq_fit(p)

best_fit, min_val = None, np.inf
"""
qs_vals = []
for i in range(1):
    for j in range(1):
        print(i, j)
        put = p0 + array([0, 0, i, j, 0])
        gof = freq_fit(put)
        qs_vals.append(gof)
        print(f"goodness of fit: {gof}")
        if gof < min_val:
            print(f"new min val: {put} ({gof})")
            min_val = gof
            best_fit = put

print(best_fit, min_val)
plt.figure("QS")
plt.plot(qs_vals)
plt.show()
"""

"""
fails0 = np.sum(np.abs(n0_truth[f0_idx:f1_idx] - n[f0_idx:f1_idx, 1]) > 0.01)
fails1 = np.sum(np.abs(n1_truth[f0_idx:f1_idx] - n[f0_idx:f1_idx, 2]) > 0.01)

print(f"fails n0: {fails0}, fails n1: {fails1}")

# plt.figure("Refractive index fitted")
# plt.plot(freqs, n0_truth, label="n0 truth")
# plt.plot(freqs, n1_truth, label="n1 truth")
# plt.legend()
"""


def calc_model_long(p_):
    freqs = np.arange(-10.0, 10.0, 0.001)
    one = np.ones_like(freqs)
    n = array([one, 1.6 * one, 2.80 * one, 1.6 * one, one]).T

    r_mod_fd_ = np.zeros((len(freqs), len(freqs)), dtype=complex)
    r_mod_fd_[:, 0] = freqs
    for f_idx, freq in enumerate(freqs):
        n_list = n[f_idx]
        lam_vac = 10 ** 6 * c0 / (freq * 10 ** 12)
        r_mod_fd_[f_idx, 1] = -1 * coh_tmm_slim("s", n_list, p_, 8 * pi / 180, lam_vac)

    return r_mod_fd_


res = 4


def calc_model(p_, n_, fast=False):
    if fast:
        r_mod_fd_ = np.zeros_like(ref_fd[::res], dtype=complex)
        r_mod_fd_[:, 0] = freqs[::res]
        n_ = n_[::res]
    else:
        r_mod_fd_ = np.zeros_like(ref_fd, dtype=complex)
        r_mod_fd_[:, 0] = freqs

    for f_idx, freq in enumerate(r_mod_fd_[:, 0].real):
        n_list = n_[f_idx]
        lam_vac = 10 ** 6 * c0 / (freq * 10 ** 12)
        r_mod_fd_[f_idx, 1] = -1 * coh_tmm_slim_unsafe("s", n_list, p_, 8 * pi / 180, lam_vac)

    return r_mod_fd_


best_fit, min_val = (0, 0), np.inf
for i in range(-10, 20):
    for j in range(-10, 20):
        continue
        put = p0 + array([0, 0, i, j, 0])
        r_mod_fd = calc_model(put, n)

        f0, f1 = 0.000, 1.500
        f0_idx, f1_idx = np.argmin(np.abs(freqs - f0)), np.argmin(np.abs(freqs - f1))

        print(i, j)
        func_val = np.sum((np.abs(r_mod_fd[f0_idx:f1_idx, 1]) - amp_meas_[f0_idx:f1_idx]) ** 2)
        print(func_val)
        if func_val < min_val:
            best_fit = (i, j)
            min_val = func_val

print(best_fit, min_val)

mod_fd = np.zeros_like(ref_fd, dtype=complex)
mod_fd[:, 0] = freqs

r_mod_fd = calc_model(p0 + array([0, 0, *best_fit, 0]), n)
mod_fd[:, 1] = r_mod_fd[:, 1] * ref_fd[:, 1]

r_mod_fd_long = calc_model_long(p0 + array([0, 0, 5, -10, 0]))

"""
mod_td = do_ifft(mod_fd, shift=0, flip=False)

r_mod_fd = zero_pad(r_mod_fd, mult=pad)
r_mod_td = do_ifft(r_mod_fd, flip=False)
r_mod_td = shift(r_mod_td, -100)
r_mod_td = filter(r_mod_td)
r_mod_td = shift(r_mod_td, 200)

# r_mod_td[:, 1] = np.roll(r_mod_td[:, 1], -1)
r_mod_td[:, 1] /= np.max(np.abs(r_mod_td[:, 1]))

R_model = np.real(mod_fd[:, 1] * np.conj(mod_fd[:, 1]))


plt.figure("Time domain transfer function")
plt.plot(t_func_td[:, 0], t_func_td[:, 1], label=f"Transfer function {sam_idx}")
plt.plot(r_mod_td[:, 0], r_mod_td[:, 1], label=f"r model")
plt.xlabel("Time (ps)")
plt.ylabel("Amplitude")
plt.legend()
"""

# d0, d1, d2 = 0, 16, -13 array([np.inf, 42.0, 641.0, 79.0, np.inf]) good fit but not at higher f
d0, d1, d2 = p0[1], p0[2], p0[3]

r_mod_fd_ = calc_model(p0, n, fast=True)
fig0, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

ax0.plot(t_func_fd[:, 0], 20 * np.log10(amp_meas_), label=f"Transfer function {sam_idx}")
amp_line, = ax0.plot(r_mod_fd_[:, 0], 20 * np.log10(np.abs(r_mod_fd_[:, 1])), lw=2, label="Model")

ax1.plot(t_func_fd[:, 0], angle_meas_, label=f"Transfer function {sam_idx}")
phase_line, = ax1.plot(r_mod_fd_[:, 0], np.angle(r_mod_fd_[:, 1]), lw=2, label="Model")

ax0.set_ylabel("Amplitude (dB)")
ax1.set_ylabel("Phase (Rad)")
ax1.set_xlabel("Frequency (THz)")

# adjust the main plot to make room for the sliders
fig0.subplots_adjust(left=0.25, bottom=0.25)

axd0 = fig0.add_axes([0.25, 0.15, 0.65, 0.03])
d0_slider = Slider(
    ax=axd0,
    label='d0 (um)',
    valmin=20,
    valmax=100,
    valinit=d0,
)

axd1 = fig0.add_axes([0.25, 0.1, 0.65, 0.03])
d1_slider = Slider(
    ax=axd1,
    label='d1 (um)',
    valmin=600,
    valmax=680,
    valinit=d1,
)

axd2 = fig0.add_axes([0.25, 0.05, 0.65, 0.03])
d2_slider = Slider(
    ax=axd2,
    label='d2 (um)',
    valmin=20,
    valmax=100,
    valinit=d2,
)

fig1, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
fig1.subplots_adjust(left=0.25, bottom=0.25)

ax0.set_ylim((1.2, 3.2))
ax0.set_ylabel("Refractive index")
ax1.set_xlabel("Frequency (THz)")

n0_line, = ax0.plot(freqs, n[:, 1].real, lw=2, label="Refractive index n0")
n1_line, = ax0.plot(freqs, n[:, 2].real, lw=2, label="Refractive index n1")
n2_line, = ax0.plot(freqs, n[:, 3].real, lw=2, label="Refractive index n2")

n0_slider_ax = fig1.add_axes([0.15, 0.15, 0.25, 0.03])
n0_slider = RangeSlider(ax=n0_slider_ax,
                        label="n0",
                        valmin=1.4,
                        valmax=2.0,
                        valinit=(1.5038, 1.5438),
                        )

n1_slider_ax = fig1.add_axes([0.15, 0.10, 0.25, 0.03])
n1_slider = RangeSlider(ax=n1_slider_ax,
                        label="n1",
                        valmin=2.7,
                        valmax=3.1,
                        valinit=(2.7725, 2.8000),
                        )

n2_slider_ax = fig1.add_axes([0.15, 0.05, 0.25, 0.03])
n2_slider = RangeSlider(ax=n2_slider_ax,
                        label="n2",
                        valmin=1.4,
                        valmax=2.0,
                        valinit=(1.5038, 1.5513),
                        )

k0_line, = ax0.plot(freqs, n[:, 1].imag, lw=2, label="Extinction coefficient k0")
k1_line, = ax0.plot(freqs, n[:, 2].imag, lw=2, label="Extinction coefficient k1")
k2_line, = ax0.plot(freqs, n[:, 3].imag, lw=2, label="Extinction coefficient k2")

k0_slider_ax = fig1.add_axes([0.60, 0.15, 0.20, 0.03])
k0_slider = RangeSlider(ax=k0_slider_ax,
                        label="k0",
                        valmin=0.000,
                        valmax=0.100,
                        valinit=(0.000, 0.001),
                        )

k1_slider_ax = fig1.add_axes([0.60, 0.10, 0.20, 0.03])
k1_slider = RangeSlider(ax=k1_slider_ax,
                        label="k1",
                        valmin=0.000,
                        valmax=0.10,
                        valinit=(0.000, 0.0245),
                        )

k2_slider_ax = fig1.add_axes([0.60, 0.05, 0.20, 0.03])
k2_slider = RangeSlider(ax=k2_slider_ax,
                        label="k2",
                        valmin=0.000,
                        valmax=0.10,
                        valinit=(0.000, 0.0018),
                        )
f0_i = np.argmin(np.abs(r_mod_fd_[:, 0].real - 0.100))
f1_i = np.argmin(np.abs(r_mod_fd_[:, 0].real - 1.600))
f_loss = r_mod_fd_[f0_i:f1_i, 0].real

loss_amp = ((np.abs(r_mod_fd_[:, 1]) - amp_meas_[::res]) ** 2)[f0_i:f1_i]
loss_angle = ((np.angle(r_mod_fd_[:, 1]) - angle_meas_[::res]) ** 2)[f0_i:f1_i]

loss_amp_text = ax1.text(-0.5, 1.00, f"Amp. loss: {np.round(np.sum(loss_amp), 3)}")
loss_angle_text = ax1.text(-0.5, 2.50, f"Phi loss: {np.round(np.sum(loss_angle), 3)}")
loss_total_text = ax1.text(-0.5, 4.00, f"Total loss: {np.round(np.sum(loss_angle) + np.sum(loss_amp), 3)}")
ax1.set_ylim((-10, 2.5))

loss_amp_line, = ax1.plot(f_loss, np.log10(loss_amp), label="Amplitude residuals")
loss_phase_line, = ax1.plot(f_loss, np.log10(loss_angle), label="Phase residuals")

ax1.legend()

def update(val):
    n = array([one, 1.6 * one, 2.80 * one, 1.6 * one, one], dtype=complex).T

    fa_idx, fe_idx = np.argmin(np.abs(freqs - 0.001)), np.argmin(np.abs(freqs - 2.000))
    n[fa_idx:fe_idx, 1] = np.linspace(n0_slider.val[0], n0_slider.val[1], fe_idx - fa_idx)
    n[fa_idx:fe_idx, 2] = np.linspace(n1_slider.val[0], n1_slider.val[1], fe_idx - fa_idx)
    n[fa_idx:fe_idx, 3] = np.linspace(n2_slider.val[0], n2_slider.val[1], fe_idx - fa_idx)

    k0 = np.linspace(k0_slider.val[0], k0_slider.val[1], len(one))
    k1 = np.linspace(k1_slider.val[0], k1_slider.val[1], len(one))
    k2 = np.linspace(k2_slider.val[0], k2_slider.val[1], len(one))

    n = n - 1j * np.array([np.zeros_like(one), k0, k1, k2, np.zeros_like(one)]).T

    k0_line.set_ydata(n[:, 1].imag)
    k1_line.set_ydata(n[:, 2].imag)
    k2_line.set_ydata(n[:, 3].imag)

    n0_line.set_ydata(n[:, 1].real)
    n1_line.set_ydata(n[:, 2].real)
    n2_line.set_ydata(n[:, 3].real)

    r_mod_fd = calc_model(array([0, d0_slider.val, d1_slider.val, d2_slider.val, 0]), n, fast=True)
    y_data_amp = 20 * np.log10(np.abs(r_mod_fd[:, 1]))
    y_data_phase = np.angle(r_mod_fd[:, 1])

    amp_line.set_ydata(y_data_amp)
    phase_line.set_ydata(y_data_phase)

    loss_amp = ((np.abs(r_mod_fd[:, 1]) - amp_meas_[::res]) ** 2)[f0_i:f1_i]
    loss_angle = ((np.angle(r_mod_fd[:, 1]) - angle_meas_[::res]) ** 2)[f0_i:f1_i]

    loss_amp_text.set_text(f"Amp. loss: {np.round(np.sum(loss_amp), 3)}")
    loss_angle_text.set_text(f"Phi loss: {np.round(np.sum(loss_angle), 3)}")
    loss_total_text.set_text(f"Total loss: {np.round(np.sum(loss_angle) + np.sum(loss_amp), 3)}")

    loss_amp_line.set_ydata(np.log10(loss_amp))
    loss_phase_line.set_ydata(np.log10(loss_angle))

    fig0.canvas.draw_idle()
    fig1.canvas.draw_idle()


k0_slider.on_changed(update)
k1_slider.on_changed(update)
k2_slider.on_changed(update)

n0_slider.on_changed(update)
n1_slider.on_changed(update)
n2_slider.on_changed(update)

d0_slider.on_changed(update)
d1_slider.on_changed(update)
d2_slider.on_changed(update)

resetax0 = fig0.add_axes([0.8, 0.010, 0.1, 0.04])
button0 = Button(resetax0, 'Reset', hovercolor='0.975')


def reset0(event):
    d0_slider.reset()
    d1_slider.reset()
    d2_slider.reset()


button0.on_clicked(reset0)

resetax1 = fig1.add_axes([0.8, 0.010, 0.1, 0.04])
button1 = Button(resetax1, 'Reset', hovercolor='0.975')


def reset1(event):
    n0_slider.reset()
    n1_slider.reset()
    n2_slider.reset()

    k0_slider.reset()
    k1_slider.reset()
    k2_slider.reset()


button1.on_clicked(reset1)

"""
plt.figure("Amplitude transfer function")
plt.plot(t_func_fd[:, 0], 20 * np.log10(amp_meas_avg), label=f"Transfer function {sam_idx}")
plt.plot(r_mod_fd[:, 0], 20 * np.log10(np.abs(r_mod_fd[:, 1])), label=f"r model")
plt.plot(r_mod_fd_long[:, 0], 20 * np.log10(np.abs(r_mod_fd_long[:, 1])), label="r model long")
plt.xlabel("Frequency (THz)")
plt.ylabel("Amplitude (dB)")
plt.legend()

ref_fd, sam_fd = load_data(sam_idx_=sam_idx, polar=True)
phase = sam_fd[:, 2] - ref_fd[:, 2]
limited_slice = np.abs(phase) <= pi


plt.figure("Phase transfer function")
plt.plot(t_func_fd[:, 0], angle_meas_avg, label=f"Transfer function {sam_idx:04}")
plt.plot(ref_fd[limited_slice, 0], phase[limited_slice], label=f"Polar form {sam_idx:04}")
plt.plot(r_mod_fd[:, 0], np.angle(r_mod_fd[:, 1]), label=f"r model")
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (rad)")
plt.legend()

plt.figure("Time domain")
# plt.plot(ref_td[:, 0], ref_td[:, 1], label=f"Reference {sam_idx}")
# plt.plot(sam_td[:, 0], sam_td[:, 1], label=f"Sample {sam_idx}")
# plt.plot(bk_gnd_td[:, 0], bk_gnd_td[:, 1], label=f"Background")
plt.plot(mod_td[:, 0], mod_td[:, 1], label=f"model")
plt.xlabel("Time (ps)")
plt.ylabel("Amplitude (nA)")
plt.legend()


plt.figure("Amplitude")
plt.plot(freqs, 20 * np.log10(np.abs(ref_fd[:, 1] - bk_fd[:, 1])), label=f"reference {sam_idx:04}")
plt.plot(freqs, 20 * np.log10(np.abs(sam_fd[:, 1] - bk_fd[:, 1])), label=f"sample {sam_idx:04}")
plt.plot(freqs, 20 * np.log10(np.abs(bk_fd[:, 1])), label=f"background")
plt.plot(freqs, 20 * np.log10(np.abs(mod_fd[:, 1])), label=f"model")
plt.xlabel("Frequency (THz)")
plt.ylabel("Amplitude (dB)")
plt.legend()


plt.figure("Phase")
plt.plot(freqs, (np.angle(ref_fd[:, 1])), label=f"reference {sam_idx:04}")
plt.plot(freqs, (np.angle(sam_fd[:, 1])), label=f"sample {sam_idx:04}")
plt.plot(freqs, (np.angle(mod_fd[:, 1])), label=f"model")
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (rad)")
plt.legend()
"""

plt.show()
