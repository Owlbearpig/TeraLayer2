import matplotlib.pyplot as plt
import numpy as np
from numpy import array, pi
from scipy.constants import c as c0
from plot_measurement import load_data
from tmm_package import coh_tmm_slim
from functions import do_ifft, filtering, window, do_fft
from scipy.optimize import shgo


def filter(data_td):
    data_td = filtering(data_td, filt_type="hp", wn=0.22, order=1)
    data_td = filtering(data_td, filt_type="lp", wn=1.92, order=4)

    return data_td


sam_idx = 12
ref_fd, sam_fd = load_data(sam_idx_=sam_idx)
freqs = ref_fd[:, 0].real
R_meas = np.abs(sam_fd[:, 1])

offset = 0.00 # 0.16
ref_td, sam_td = do_ifft(ref_fd, shift=100-offset), do_ifft(sam_fd, shift=100)

ref_td, sam_td = filter(ref_td), filter(sam_td)

ref_td = window(ref_td, win_width=36, en_plot=False, slope=0.3)
sam_td = window(sam_td, win_width=36, en_plot=False, slope=0.3)

ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

t_func_fd = np.zeros_like(ref_fd, dtype=complex)
t_func_fd[:, 0] = freqs
t_func_fd[:, 1] = sam_fd[:, 1] / ref_fd[:, 1]

t_func_td = do_ifft(t_func_fd, shift=100)
t_func_td = filter(t_func_td)
t_func_td = window(t_func_td, win_start=78, win_width=36, en_plot=False, slope=0.3)

t_func_fd = do_fft(t_func_td, shift=-100)

f0, f1 = 0.25, 1.75
f0_idx, f1_idx = np.argmin(np.abs(freqs - f0)), np.argmin(np.abs(freqs - f1))

np.random.seed(420)

bounds = ((1.45, 1.55), (2.75, 2.85))
p_sol = array([np.inf, 43.0, 641.0, 74.0, np.inf])
one = np.ones_like(freqs)

#n0_truth = np.random.uniform(*bounds[0], len(freqs))
#n1_truth = np.random.uniform(*bounds[1], len(freqs))
n0_truth = 1.7*one
n1_truth = 2.8*one
n_truth = array([one, n0_truth, n1_truth, n0_truth, one]).T
#n_truth = array([one, 1.692027738932626*one, 2.884941014415432*one, 1.692027738932626*one, one]).T
#n_truth = array([one, 1.7*one, 2.8*one, 1.7*one, one]).T

for f_idx in range(len(freqs)):
    lam_vac = 10 ** 6 * c0 / (freqs[f_idx] * 10 ** 12)
    # t_func_fd[f_idx, 1] = -1 * coh_tmm_slim("s", n_truth[f_idx], p_sol, 8 * pi / 180, lam_vac)


def freq_fit():
    def cost(p, f_idx):
        n_list = array([1, p[0], p[1], p[0], 1])
        #n_list = array([1, 1.5, p[0], 1.5, 1])

        lam_vac = 10 ** 6 * c0 / (freqs[f_idx] * 10 ** 12)
        mod_fd = -1 * coh_tmm_slim("s", n_list, p_sol, 8 * pi / 180, lam_vac)

        loss_phi = (np.angle(t_func_fd[f_idx, 1]) - np.angle(mod_fd)) ** 2
        loss_amp = (np.abs(t_func_fd[f_idx, 1]) - np.abs(mod_fd)) ** 2

        return loss_phi + loss_amp

    #x, y = np.linspace(1.69201, 1.69203, 200), np.linspace(2.88493, 2.88495, 200)
    x, y = np.linspace(1.5, 2, 200), np.linspace(2.5, 3.5, 200)
    x_long, y_long = np.linspace(1.5, 3, 1000), np.linspace(2.5, 3.5, 1000)

    plt.figure()
    """
    Frequency 1.4629999999999999 (1697/4500)
    Result: [1.6875,    2.8852679] (3.2237342661786996e-05)
    truth: [1.692027738932626, 2.884941014415432]
    """
    f_idx_ = 1697 - 600
    print(freqs[f_idx_])
    plt.plot(x_long, [cost([x_, 2.91], f_idx_) for x_ in x_long])
    plt.plot(x_long, [cost([x_, 2.9], f_idx_) for x_ in x_long])
    plt.plot(y_long, [cost([1.7, y_], f_idx_) for y_ in y_long])
    plt.plot(y_long, [cost([1.6, y_], f_idx_) for y_ in y_long])

    plt.figure()
    grid = np.zeros((len(x), len(y)))
    for i, p0 in enumerate(x):
        for j, p1 in enumerate(y):
            grid[i, j] = (cost([p0, p1], 1697))
    plt.imshow(grid, extent=[x[0], x[-1], y[0], y[-1]], origin="lower")
    #plt.show()
    print(cost([1.692,    2.884941014415432], 1697))

    n = array([one, 1.5 * one, 2.80 * one, 1.5 * one, one]).T
    for f_idx, freq in enumerate(freqs):
        if (freq < f0) or (freq > f1):
            continue
        print(f"Frequency {freq} ({f_idx}/{len(freqs)})")
        iters = 3
        res = shgo(cost, args=(f_idx,), bounds=bounds, iters=iters, n=2**8)
        while res.fun > 1e-10:
            iters += 1
            if iters == 5:
                break
            res = shgo(cost, args=(f_idx,), bounds=bounds, iters=iters, n=2**12)

        print(f"Result: {res.x} ({res.fun})")
        print(f"truth: [{n0_truth[f_idx]}, {n1_truth[f_idx]}]")
        if res.fun < 1e-10:
            n[f_idx, 1], n[f_idx, 2], n[f_idx, 3] = res.x[0], res.x[1], res.x[0],
        else:
            n[f_idx, 1], n[f_idx, 2], n[f_idx, 3] = 0, 0, 0
        # n[f_idx, 1], n[f_idx, 2], n[f_idx, 3] = 1.5, res.x[0], 1.5,

    plt.figure("Refractive index fitted")
    plt.plot(freqs, n[:, 1], label="n0")
    plt.plot(freqs, n[:, 2], label="n1")
    plt.plot(freqs, n[:, 3], label="n2")
    plt.axvline(x=f0, color="b")
    plt.axvline(x=f1, color="b", label="Fit range")
    plt.legend()
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive index")

    return n


def fit():
    dx = freqs[f1_idx] - freqs[f0_idx]

    def linear_n(p):
        # y0, y1 -> a = (y1 - y0) / dx, y0 - a * x0 = b, n = a * x + b
        # (x0, x1 = 1.25 THz - 1.75 THz)

        """
        a0, a1, a2 = (p[1] - p[0]) / dx, (p[3] - p[2]) / dx, (p[5] - p[4]) / dx
        b0, b1, b2 = p[0] - a0 * f0, p[2] - a0 * f0, p[4] - a0 * f0
        n0, n1, n2 = a0 * freqs + b0, a1 * freqs + b1, a2 * freqs + b2
        """

        a0, a1, a2 = (p[1] - p[0]) / dx, (p[3] - p[2]) / dx, (p[1] - p[0]) / dx
        b0, b1, b2 = p[0] - a0 * f0, p[2] - a1 * f0, p[0] - a2 * f0
        n0, n1, n2 = a0 * freqs + b0, a1 * freqs + b1, a2 * freqs + b2

        n = array([one, n0, n1, n2, one]).T

        return n

    def cost(p):
        n = linear_n(p)

        mod_fd = np.zeros_like(freqs, dtype=complex)
        for f_idx, freq in enumerate(freqs):
            n_list = n[f_idx]
            lam_vac = 10 ** 6 * c0 / (freq * 10 ** 12)
            mod_fd[f_idx] = -1 * coh_tmm_slim("s", n_list, p_sol, 8 * pi / 180, lam_vac) * ref_fd[f_idx, 1]

        loss = np.sum((np.angle(sam_fd[f0_idx:f1_idx, 1]) - np.angle(mod_fd[f0_idx:f1_idx])) ** 2)

        return loss / (f1_idx - f0_idx)

    bounds = ((1.5, 1.55), (1.55, 1.6),
              (2.7, 2.8), (2.8, 2.9))

    # res = shgo(cost, bounds, iters=1)
    # print(res)

    p = [1.55, 1.6, 2.7, 2.9]
    n = linear_n(p)

    plt.figure("Refractive index fitted")
    plt.plot(freqs, n[:, 1], label="n0")
    plt.plot(freqs, n[:, 2], label="n1")
    plt.plot(freqs, n[:, 3], label="n2")
    plt.axvline(x=f0, color="b")
    plt.axvline(x=f1, color="b", label="Fit range")
    plt.legend()
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive index")

    return n

n = array([one, 1.6 * one, 2.80 * one, 1.6 * one, one]).T
# n = freq_fit()

fails0 = np.sum(np.abs(n0_truth[f0_idx:f1_idx] - n[f0_idx:f1_idx, 1]) > 0.01)
fails1 = np.sum(np.abs(n1_truth[f0_idx:f1_idx] - n[f0_idx:f1_idx, 2]) > 0.01)

print(f"fails n0: {fails0}, fails n1: {fails1}")

plt.figure("Refractive index fitted")
plt.plot(freqs, n0_truth, label="n0 truth")
plt.plot(freqs, n1_truth, label="n1 truth")
plt.legend()

r_mod_fd, mod_fd = np.zeros_like(ref_fd, dtype=complex), np.zeros_like(ref_fd, dtype=complex)
r_mod_fd[:, 0], mod_fd[:, 0] = freqs, freqs
for f_idx, freq in enumerate(freqs):
    n_list = n[f_idx]
    lam_vac = 10 ** 6 * c0 / (freq * 10 ** 12)
    r_mod_fd[f_idx, 1] = -1 * coh_tmm_slim("s", n_list, p_sol, 8 * pi / 180, lam_vac)
    mod_fd[f_idx, 1] = r_mod_fd[f_idx, 1] * ref_fd[f_idx, 1]

mod_td = do_ifft(mod_fd, shift=0)
r_mod_td = do_ifft(r_mod_fd, shift=100)

R_model = np.real(mod_fd[:, 1] * np.conj(mod_fd[:, 1]))

plt.figure("Time domain transfer function")
plt.plot(t_func_td[:, 0], t_func_td[:, 1], label=f"Transfer function {sam_idx}")
plt.plot(r_mod_td[:, 0], r_mod_td[:, 1], label=f"r model")
plt.xlabel("Time (ps)")
plt.ylabel("Amplitude (nA)")
plt.legend()

plt.figure("Amplitude transfer function")
plt.plot(freqs, 20 * np.log10(np.abs(t_func_fd[:, 1])), label=f"Transfer function {sam_idx}")
plt.plot(freqs, 20 * np.log10(np.abs(r_mod_fd[:, 1])), label=f"r model")
plt.xlabel("Frequency (THz)")
plt.ylabel("Amplitude (dB)")
plt.legend()

plt.figure("Phase transfer function")
plt.plot(freqs, np.angle(t_func_fd[:, 1]), label=f"Transfer function {sam_idx:04}")
plt.plot(freqs, np.angle(r_mod_fd[:, 1]), label=f"r model")
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
plt.plot(freqs, np.unwrap(np.angle(ref_fd[:, 1])), label=f"reference {sam_idx:04}")
plt.plot(freqs, np.unwrap(np.angle(sam_fd[:, 1])), label=f"sample {sam_idx:04}")
plt.plot(freqs, np.unwrap(np.angle(mod_fd[:, 1])), label=f"model")
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (rad)")
plt.legend()

plt.show()
