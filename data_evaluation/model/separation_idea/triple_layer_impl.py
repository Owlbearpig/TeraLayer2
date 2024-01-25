import numpy as np
from tmm import list_snell, interface_r
from numpy import conj, abs, cos
from meas_eval.consts import thea, c_thz
from meas_eval.JumpingLaser.parse_data import Measurement, ModelMeasurement
import matplotlib.pyplot as plt
from model.tmm_package import coh_tmm_slim

num_layers = 5  # first and last layers are air
pol = "s"


def triple_layer_impl(sam_meas_: Measurement):
    freqs = sam_meas_.freq

    n_list = sam_meas_.sample.value.get_ref_idx(freqs)
    r_exp_ = sam_meas_.r_avg
    lam_vac = c_thz / freqs

    r_fn = np.zeros((len(freqs), num_layers, num_layers), dtype=complex)
    kz_list, th_list = np.zeros((2, len(freqs), num_layers), dtype=complex)
    for freq_idx_ in range(freqs.size):
        th_list[freq_idx_] = list_snell(n_list[freq_idx_], thea).T
        kz_list[freq_idx_, :] = 2 * np.pi * n_list[freq_idx_] * cos(th_list[freq_idx_]) / lam_vac[freq_idx_]
        for i in range(num_layers - 1):
            r_fn[freq_idx_, i, i + 1] = interface_r(pol, n_list[freq_idx_, i], n_list[freq_idx_, i + 1],
                                                    th_list[freq_idx_, i], th_list[freq_idx_, i + 1])

    r = -np.abs(r_exp_)
    u = np.exp(1j * np.angle(r_exp_))

    c0 = r * r_fn[:, 0, 1] * r_fn[:, 3, 4] * u - r_fn[:, 3, 4]
    c1 = r * r_fn[:, 1, 2] * r_fn[:, 3, 4] * u - r_fn[:, 0, 1] * r_fn[:, 1, 2] * r_fn[:, 3, 4]
    c2 = (r * r_fn[:, 0, 1] * r_fn[:, 1, 2] * r_fn[:, 2, 3] * u - r_fn[:, 1, 2] * r_fn[:, 2, 3]) * r_fn[:, 3, 4]
    c3 = r * r_fn[:, 0, 1] * r_fn[:, 2, 3] * u - r_fn[:, 2, 3]
    c4 = r * r_fn[:, 2, 3] * r_fn[:, 3, 4] * u - r_fn[:, 0, 1] * r_fn[:, 2, 3] * r_fn[:, 3, 4]
    c5 = r * r_fn[:, 1, 2] * r_fn[:, 2, 3] * u - r_fn[:, 0, 1] * r_fn[:, 1, 2] * r_fn[:, 2, 3]
    c6 = r * r_fn[:, 0, 1] * r_fn[:, 1, 2] * u - r_fn[:, 1, 2]
    c7 = r * u - r_fn[:, 0, 1]

    def expr1_(d1_, d2_, freq_idx_=0):
        phi0 = d1_ * kz_list[freq_idx_, 1]
        x_ = np.exp(1j * 2 * phi0)
        phi1 = d2_ * kz_list[freq_idx_, 2]
        y_ = np.exp(1j * 2 * phi1)

        num = c0[freq_idx_] + c1[freq_idx_] * x_ + c2[freq_idx_] * y_ + c4[freq_idx_] * x_ * y_
        den = c3[freq_idx_] + c5[freq_idx_] * x_ + c6[freq_idx_] * y_ + c7[freq_idx_] * x_ * y_

        s = np.abs(num / den)

        return (1 - s) ** 2

    d1_truth, d2_truth, d3_truth = sam_meas_.sample.value.thicknesses.astype(int)

    d1 = np.arange(max(0, d1_truth - 100), d1_truth + 100, 1)
    d2 = np.arange(max(0, d2_truth - 100), d2_truth + 100, 1)
    d3 = np.arange(max(0, d3_truth - 100), d3_truth + 100, 1)

    X, Y = np.meshgrid(d1, d2)

    expr1_sum = np.sum([expr1_(X, Y, f_idx) for f_idx in range(len(freqs))], axis=0)

    i, j = np.unravel_index(np.argmin(expr1_sum), expr1_sum.shape)
    d1_found, d2_found = d1[j], d2[i]
    print(np.min(expr1_sum), f"Found (global)minimum at d1: {d1_found} um, d2: {d2_found} um")

    plt.figure(f"triple layer eval")
    plt.scatter(*(d1_found, d2_found), s=20, c='red', marker='x')
    plt.title(f"Freq. sum expr1")
    plt.imshow(expr1_sum,
               extent=[d1[0], d1[-1], d2[0], d2[-1]],
               origin="lower",
               # interpolation='bilinear',
               # cmap="plasma",
               vmin=0, vmax=np.mean(expr1_sum),
               )
    s = f"({d1_found}, {d2_found})"
    plt.text(d1_found+5, d2_found+5, s, fontsize=14, color="red")
    plt.xlabel("$d_1$")
    plt.ylabel("$d_2$")

    plt.figure()
    plt.title("(full model - measurement)$^2$, wrt $d_3$")
    r_exp_mod = np.zeros_like(freqs, dtype=complex)
    y_vals = []
    best_fit = (None, np.inf)
    for d3_ in d3:
        for freq_idx_ in range(len(freqs)):
            d = np.array([np.inf, d1_found, d2_found, d3_, np.inf], dtype=float)
            r_exp_mod[freq_idx_] = -coh_tmm_slim(pol, n_list[freq_idx_], d, thea, lam_vac[freq_idx_])
        err = np.sum((r_exp_mod.real - r_exp_.real) ** 2 + (r_exp_mod.imag - r_exp_.imag) ** 2)
        y_vals.append(err)
        if err < best_fit[1]:
            best_fit = (d3_, err)

    plt.plot(d3, y_vals, label="Squared differences")
    min_point = (d3[np.argmin(y_vals)], np.min(y_vals))
    plt.annotate(f"{min_point[0]}, {min_point[1]}", xy=(min_point[0], min_point[1]), xytext=(-20, 20),
                 textcoords='offset points', ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                 color='red'))
    plt.xlabel("$d_3$")
    plt.ylabel("Loss")

    return expr1_sum

