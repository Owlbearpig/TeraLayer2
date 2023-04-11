import matplotlib.pyplot as plt
import numpy as np
from numpy import array, pi
from scipy.constants import c as c0
from plot_measurement import load_data
from tmm_package import coh_tmm_slim
from functions import do_ifft, filtering, window, do_fft, zero_pad, shift
from scipy.optimize import shgo


def filter(data_td, en=True):
    if not en:
        return data_td

    data_td = filtering(data_td, filt_type="hp", wn=0.22, order=1)
    # data_td = filtering(data_td, filt_type="lp", wn=1.76, order=4)
    data_td = filtering(data_td, filt_type="lp", wn=2.10, order=7)

    return data_td


shift_ = 0

# offset = 0.111
# offset = 0.03
offset = 0  # 0.80
pad = 8

angle_meas, amp_meas = [], []
for sam_idx in range(101):
    # sam_idx = 23
    ref_fd, sam_fd = load_data(sam_idx_=sam_idx)
    freqs = ref_fd[:, 0].real
    R_meas = np.abs(sam_fd[:, 1])

    ref_td, sam_td = do_ifft(ref_fd), do_ifft(sam_fd)

    # ref_td, sam_td = filter(ref_td), filter(sam_td)

    ref_td, sam_td = shift(ref_td, 100 - offset), shift(sam_td, 100)

    ref_td = window(ref_td, win_width=250, win_start=0, en_plot=False, slope=0.03, label="Ref")
    sam_td = window(sam_td, win_width=250, win_start=0, en_plot=False, slope=0.03, label="Sam")

    ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

    t_func_fd = np.zeros_like(ref_fd, dtype=complex)
    t_func_fd[:, 0] = freqs
    t_func_fd[:, 1] = sam_fd[:, 1] / ref_fd[:, 1]

    # t_func_fd = zero_pad(t_func_fd, mult=pad)

    freqs = ref_fd[:, 0].real

    """
    plt.figure("test")
    plt.plot(np.angle(t_func_fd), label="angle before")
    plt.figure("test2")
    plt.plot(np.abs(t_func_fd), label="magn before")
    """

    t_func_td = do_ifft(t_func_fd, flip=False)
    t_func_td = shift(t_func_td, -100)
    t_func_td = filter(t_func_td)
    t_func_td = shift(t_func_td, 100)
    # t_func_td = window(t_func_td, win_start=shift-5, win_width=25, en_plot=False, slope=0.3)
    dt = np.mean(np.diff(t_func_td[:, 0]))

    t_func_fd = do_fft(t_func_td)
    t_func_td = shift(t_func_td, 100)

    angle_meas.append(np.angle(t_func_fd[:, 1]))
    amp_meas.append(np.abs(t_func_fd[:, 1]))

angle_meas_all = array(angle_meas)
amp_meas_all = array(amp_meas)

angle_meas_avg = np.mean(array(angle_meas), axis=0)
amp_meas_avg = np.mean(array(amp_meas), axis=0)

plt.figure()
plt.plot(freqs, angle_meas)
plt.plot(freqs, amp_meas)
plt.show()

"""
plt.figure("test")
plt.plot(np.angle(t_func_fd), label="angle after")
plt.legend()
plt.figure("test2")
plt.plot(np.abs(t_func_fd), label="magn after")
plt.legend()
plt.show()
"""

f0, f1 = 0.220, 2.200
f0_idx, f1_idx = np.argmin(np.abs(freqs - f0)), np.argmin(np.abs(freqs - f1))

np.random.seed(420)

bounds = ((2.75, 2.90), (1.50, 1.60))
bounds = ((2.80, 2.90), (1.50, 1.70))
# bounds = ((3.00, 3.10), )
# bounds = ((1.55, 1.65), )

p_sol = array([np.inf, 46.0, 641.0, 79.0, np.inf])
one = np.ones_like(freqs)

# n0_truth = np.random.uniform(*bounds[0], len(freqs))
# n1_truth = np.random.uniform(*bounds[1], len(freqs))
n0_truth = 1.65 * one
n1_truth = 2.86 * one
n_truth = array([one, n0_truth, n1_truth, n0_truth, one]).T
# n_truth = array([one, 1.692027738932626*one, 2.884941014415432*one, 1.692027738932626*one, one]).T
# n_truth = array([one, 1.7*one, 2.8*one, 1.7*one, one]).T

for f_idx in range(len(freqs)):
    lam_vac = 10 ** 6 * c0 / (freqs[f_idx] * 10 ** 12)
    # t_func_fd[f_idx, 1] = -1 * coh_tmm_slim("s", n_truth[f_idx], p_sol, 8 * pi / 180, lam_vac)

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
        n_list = array([1, 1.65, p[0], p[1], 1])
        # n_list = array([1, 1.6, p[0], p[1], 1])
        # n_list = array([1, p[0], 2.8])

        lam_vac = 10 ** 6 * c0 / (freqs[f_idx] * 10 ** 12)
        mod_fd = -1 * coh_tmm_slim("s", n_list, thicknesses, 8 * pi / 180, lam_vac)

        loss_phi = (angle_meas[f_idx] - np.angle(mod_fd)) ** 2
        loss_amp = (amp_meas[f_idx] - np.abs(mod_fd)) ** 2

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
    n1_res, n2_res = [], []
    n = array([one, 1.6 * one, 2.80 * one, 1.6 * one, one]).T
    for f_idx, freq in enumerate(freqs):
        if (freq < f0) or (freq > f1) or (f_idx % 4 != 0):
        #if (freq < f0) or (freq > f1):
            continue
        print(f"Frequency {freq} ({f_idx}/{len(freqs)})")
        iters = 5
        res = shgo(cost, args=(f_idx,), bounds=bounds, iters=iters, n=2 ** 8)
        while res.fun > 1e-10:
            iters += 1
            if iters == 7:
                break
            res = shgo(cost, args=(f_idx,), bounds=bounds, iters=iters, n=2 ** 12)

        print(f"Result: {res.x} ({res.fun})")
        # print(f"truth: [{n0_truth[f_idx]}, {n1_truth[f_idx]}]")
        n[f_idx, 2], n[f_idx, 3] = res.x[0], res.x[1]
        n1_res.append(res.x[0]), n2_res.append(res.x[1])

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
    gof = np.mean(np.abs(np.fft.rfft(n1_res))[1:])
    gof = np.max(np.abs(np.fft.rfft(n1_res-np.mean(n1_res)))[1:])


    # gof = np.std(n1_res) + np.std(n2_res)

    return gof


n = array([one, 1.6 * one, 2.80 * one, 1.6 * one, one]).T

best_fit, min_val = None, np.inf
p0 = array([np.inf, 46.0, 651.0, 69.0, np.inf])  # truth: array([np.inf, 46.0, 641.0, 79.0, np.inf])
p0 = array([np.inf, 46.0, 641.0, 79.0, np.inf])
qs_vals = []
for i in range(1):
    for j in range(20):
        print(i, j)
        put = p0 + array([0, 0, i, j, 0])
        gof = 0 # freq_fit(put)
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

fails0 = np.sum(np.abs(n0_truth[f0_idx:f1_idx] - n[f0_idx:f1_idx, 1]) > 0.01)
fails1 = np.sum(np.abs(n1_truth[f0_idx:f1_idx] - n[f0_idx:f1_idx, 2]) > 0.01)

print(f"fails n0: {fails0}, fails n1: {fails1}")

# plt.figure("Refractive index fitted")
# plt.plot(freqs, n0_truth, label="n0 truth")
# plt.plot(freqs, n1_truth, label="n1 truth")
# plt.legend()

r_mod_fd, mod_fd = np.zeros_like(ref_fd, dtype=complex), np.zeros_like(ref_fd, dtype=complex)
r_mod_fd[:, 0], mod_fd[:, 0] = freqs, freqs
for f_idx, freq in enumerate(freqs):
    n_list = n[f_idx]
    lam_vac = 10 ** 6 * c0 / (freq * 10 ** 12)
    r_mod_fd[f_idx, 1] = -1 * coh_tmm_slim("s", n_list, p_sol, 8 * pi / 180, lam_vac)
    mod_fd[f_idx, 1] = r_mod_fd[f_idx, 1] * ref_fd[f_idx, 1]

mod_td = do_ifft(mod_fd, shift=0, flip=False)

r_mod_fd = zero_pad(r_mod_fd, mult=pad)
r_mod_td = do_ifft(r_mod_fd, flip=True)
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

plt.figure("Amplitude transfer function")
plt.plot(t_func_fd[:, 0], 20 * np.log10(amp_meas), label=f"Transfer function {sam_idx}")
plt.plot(r_mod_fd[:, 0], 20 * np.log10(np.abs(r_mod_fd[:, 1])), label=f"r model")
plt.xlabel("Frequency (THz)")
plt.ylabel("Amplitude (dB)")
plt.legend()

ref_fd, sam_fd = load_data(sam_idx_=sam_idx, polar=True)
phase = sam_fd[:, 2] - ref_fd[:, 2]
limited_slice = np.abs(phase) <= pi

plt.figure("Phase transfer function")
plt.plot(t_func_fd[:, 0], angle_meas, label=f"Transfer function {sam_idx:04}")
plt.plot(ref_fd[limited_slice, 0], phase[limited_slice], label=f"Polar form {sam_idx:04}")
plt.plot(r_mod_fd[:, 0], np.angle(r_mod_fd[:, 1]), label=f"r model")
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (rad)")
plt.legend()

plt.figure("Time domain")
plt.plot(ref_td[:, 0], ref_td[:, 1], label=f"Reference {sam_idx}")
plt.plot(sam_td[:, 0], sam_td[:, 1], label=f"Sample {sam_idx}")
# plt.plot(bk_gnd_td[:, 0], bk_gnd_td[:, 1], label=f"Background")
plt.plot(mod_td[:, 0], mod_td[:, 1], label=f"model")
plt.xlabel("Time (ps)")
plt.ylabel("Amplitude (nA)")
plt.legend()

plt.figure("Amplitude")
plt.plot(freqs, 20 * np.log10(np.abs(ref_fd[:, 1])), label=f"reference {sam_idx:04}")
plt.plot(freqs, 20 * np.log10(np.abs(sam_fd[:, 1])), label=f"sample {sam_idx:04}")
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

plt.show()
