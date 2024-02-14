import numpy as np
from tmm import list_snell, interface_r
from numpy import conj, abs, cos
from meas_eval.consts import thea, c_thz
from meas_eval.JumpingLaser.parse_data import Measurement, ModelMeasurement, SystemEnum, SamplesEnum
import matplotlib.pyplot as plt
from model.tmm_package import coh_tmm_slim
from functions import std_err
from scipy.optimize import minimize


def triple_layer_impl(sam_meas_: Measurement, ts_meas_: Measurement, mod_meas_: ModelMeasurement, selected_sweep_):
    num_layers = 5  # first and last layers are air
    pol = "s"

    freqs = sam_meas_.freq
    ts_freq_idx = [np.argmin(np.abs(freq - mod_meas_.freq)) for freq in freqs]

    is_ts_meas = sam_meas_.system.name == SystemEnum.TSweeper.name
    if is_ts_meas:
        freq_min, freq_max = 0.2, 1.25
        freq_min_idx, freq_max_idx = np.argmin(np.abs(freq_min - freqs)), np.argmin(np.abs(freq_max - freqs))
        f_res = 10
        freqs = sam_meas_.freq[freq_min_idx:freq_max_idx:f_res]

    d1_truth, d2_truth, d3_truth = sam_meas_.sample.value.thicknesses.astype(int)

    d_truth = sam_meas_.sample.value.thicknesses.astype(int)
    bounds = []
    for i in range(3):
        min_val = max(20, d_truth[i] - 100)
        if d_truth[i] - 100 < 0:
            max_val = 200
        else:
            max_val = d_truth[i] + 100
        bounds.append((min_val, max_val))

    d1 = np.arange(bounds[0][0], bounds[0][1] + 1, 1)
    d2 = np.arange(bounds[1][0], bounds[1][1] + 1, 1)
    d3 = np.arange(bounds[2][0], 150 + 1, 1)

    if is_ts_meas:
        real_weights, imag_weights = np.ones((2, len(freqs)))
        weights = np.ones(len(freqs))
    else:
        real_weights, imag_weights = 1 / np.std(sam_meas_.r.real, axis=0), 1 / np.std(sam_meas_.r.imag, axis=0)
        real_weights, imag_weights = 4*real_weights, imag_weights
        # real_weights, imag_weights = np.ones((2, len(freqs)))
        # real_weights = np.array([1, 1, 1, 1, 1, 0])
        # imag_weights = np.array([1, 1, 1, 1, 1, 0])
        weights = np.ones(len(freqs))

    def eval_sample(sweep_idx=None):
        if sweep_idx is None:
            r_exp_ = sam_meas_.r_avg
        else:
            r_exp_ = sam_meas_.r[sweep_idx]
        if is_ts_meas:
            r_exp_ = sam_meas_.r_avg[freq_min_idx:freq_max_idx:f_res]

        if (sam_meas_.sample == SamplesEnum.ampelMannLeft) and not is_ts_meas:
            pass
            # print(mod_meas_.freq[ts_freq_idx[2]], mod_meas_.freq[ts_freq_idx[5]])
            # r_exp_[0] = mod_meas_.r_avg[ts_freq_idx[0]]
            # r_exp_[1] = mod_meas_.r_avg[ts_freq_idx[1]]
            # r_exp_[2] = mod_meas_.r_avg[ts_freq_idx[2]]  ##
            # r_exp_[3] = mod_meas_.r_avg[ts_freq_idx[3]]
            # r_exp_[4] = mod_meas_.r_avg[ts_freq_idx[4]]
            # r_exp_[5] = mod_meas_.r_avg[ts_freq_idx[5]]  ##
            # r_exp_[5] = mod_meas_.r_avg[ts_freq_idx[5]].real + 1j*r_exp_[5].imag  ##

        if (sam_meas_.sample == SamplesEnum.ampelMannRight) and not is_ts_meas:
            pass
            # print(print(mod_meas_.freq[ts_freq_idx[2]], mod_meas_.freq[ts_freq_idx[4]], mod_meas_.freq[ts_freq_idx[5]]))
            # r_exp_ = mod_meas_.r_avg[ts_freq_idx]
            # r_exp_[0] = mod_meas_.r_avg[ts_freq_idx[0]]
            # r_exp_[1] = mod_meas_.r_avg[ts_freq_idx[1]]
            # r_exp_[2] = mod_meas_.r_avg[ts_freq_idx[2]] ##
            # r_exp_[3] = mod_meas_.r_avg[ts_freq_idx[3]]
            # r_exp_[4] = mod_meas_.r_avg[ts_freq_idx[4]] ##
            # r_exp_[5] = mod_meas_.r_avg[ts_freq_idx[5]] ##

        n_list = sam_meas_.sample.value.get_ref_idx(freqs)
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

        X, Y = np.meshgrid(d1, d2)

        expr1_err = np.array([expr1_(X, Y, f_idx) for f_idx in range(len(freqs))])
        expr1_err = (np.multiply(expr1_err.T, weights)).T
        expr1_sum_ = np.sum(expr1_err, axis=0)

        i, j = np.unravel_index(np.argmin(expr1_sum_), expr1_sum_.shape)
        d1_found_, d2_found_ = d1[j], d2[i]

        def fun(p):
            r_exp_mod = np.zeros_like(freqs, dtype=complex)
            for freq_idx_ in range(len(freqs)):
                d = np.array([np.inf, *p, np.inf], dtype=float)
                r_exp_mod[freq_idx_] = -coh_tmm_slim(pol, n_list[freq_idx_], d, thea, lam_vac[freq_idx_])
            real_err = (r_exp_mod.real - r_exp_.real) ** 2
            imag_err = (r_exp_mod.imag - r_exp_.imag) ** 2

            tot_err = np.sum(real_err * real_weights + imag_err * imag_weights)

            return tot_err

        if sweep_idx == selected_sweep_:
            best_res = minimize(fun, np.array([d1_found_, d2_found_, d3[0]]))
            for d3_ in [50, 70]:
                x0 = np.array([d1_found_, d2_found_, d3_])
                res = minimize(fun, x0)
                print(res)
                if res.fun < best_res.fun:
                    best_res = res

            print("BEST RES: ", best_res)

        r_exp_mod = np.zeros_like(freqs, dtype=complex)
        d3_err_ = []
        best_fit = (None, np.inf)
        for d3_ in d3:
            for freq_idx_ in range(len(freqs)):
                d = np.array([np.inf, d1_found_, d2_found_, d3_, np.inf], dtype=float)
                r_exp_mod[freq_idx_] = -coh_tmm_slim(pol, n_list[freq_idx_], d, thea, lam_vac[freq_idx_])
            real_err = (r_exp_mod.real - r_exp_.real) ** 2
            imag_err = (r_exp_mod.imag - r_exp_.imag) ** 2

            tot_err = np.sum(real_err * real_weights + imag_err * imag_weights)
            d3_err_.append(tot_err)
            if tot_err < best_fit[1]:
                best_fit = (d3_, tot_err)

        d3_found_ = d3[np.argmin(d3_err_)]

        print(f"Sweep: {sweep_idx}")
        print(f"Found minimum {np.min(expr1_sum_)} at (d1: {d1_found_}, d2: {d2_found_}, d3: {d3_found_}) um")

        if sweep_idx == selected_sweep_:
            print(d1_found_, d2_found_, d3_found_)
            print(best_res)
            exit()

        return d1_found_, d2_found_, d3_found_, expr1_sum_, d3_err_

    sweep_s = "alle sweeps" if selected_sweep_ is None else f"sweep {selected_sweep_}"
    if is_ts_meas:
        sweep_s = ""

    d1_found, d2_found, d3_found, expr1_sum, d3_err = eval_sample(selected_sweep_)

    fig_num = (str(sam_meas_.sample.name) + f"_expr1_sum_sweep_{selected_sweep_}")
    s = "TSweeper" if is_ts_meas else "PIC"
    if not plt.fignum_exists(fig_num):
        fig, (ax0, ax1) = plt.subplots(1, 2, num=fig_num)
    else:
        fig = plt.figure(fig_num)
        ax0, ax1 = fig.get_axes()
    ax = ax1 if is_ts_meas else ax0
    ax.set_title("$\log_{10}$" + f"(Residuum) {s} {sweep_s}")
    vmin = np.min(np.log10(expr1_sum))
    vmax = np.mean(np.log10(expr1_sum))
    ax.imshow(np.log10(expr1_sum),
              extent=[d1[0], d1[-1], d2[0], d2[-1]],
              origin="lower",
              # interpolation='bilinear',
              # cmap="plasma",
              vmin=vmin, vmax=vmax,
              )
    s = f"({d1_found}, {d2_found}) PIC"
    if is_ts_meas:
        s = s.replace("PIC", "TSweeper")
    ax.text(d1_found + 5, d2_found + 5, s, fontsize=18, c="red")
    ax.scatter(*(d1_found, d2_found), s=25, c="red", marker='x')
    ax.set_xlabel("Dicke erste Schicht ($\mu$m)")
    ax.set_ylabel("Dicke zweite Schicht ($\mu$m)")

    # plt.figure(str(sam_meas_.sample.name) + f"_avg_{sam_meas_.system.name}")
    plt.figure(str(sam_meas_.sample.name) + f"_sweep_{selected_sweep_}")
    plt.title(f"Residuum {sweep_s}")
    plt.plot(d3, np.log10(d3_err), label=f"Residuen {sam_meas_.system.name} {sweep_s}")
    min_point = (d3_found, np.log10(np.min(d3_err)))
    plt.annotate(f"{min_point[0]}, {np.round(min_point[1], 3)}", xy=(min_point[0], min_point[1]),
                 xytext=(-20, 20),
                 textcoords='offset points', ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                 color='red', mutation_scale=22))
    plt.xlabel("Dicke dritter Schicht ($\mu$m)")
    plt.ylabel("$\log_{10}$(Residuum)")

    # return
    if is_ts_meas:
        return

    n_sweeps = sam_meas_.n_sweeps
    sweeps = list(range(n_sweeps))
    results_d1, results_d2, results_d3 = np.zeros((3, n_sweeps), dtype=float)
    min_losses = np.zeros((2, n_sweeps), dtype=float)
    for sweep_idx in sweeps:
        d1_found, d2_found, d3_found, expr1_sum, d3_err = eval_sample(sweep_idx)

        results_d1[sweep_idx], results_d2[sweep_idx], results_d3[sweep_idx] = d1_found, d2_found, d3_found
        min_losses[:, sweep_idx] = np.min(expr1_sum), np.min(d3_err)

    mean_d1, mean_d2, mean_d3 = np.mean(results_d1), np.mean(results_d2), np.mean(results_d3)
    std_d1 = np.round(std_err(results_d1), 2)
    std_d2 = np.round(std_err(results_d2), 2)
    std_d3 = np.round(std_err(results_d3), 2)

    fig, (ax0, ax1) = plt.subplots(1, 2, num=str(sam_meas_.sample.name) + "_single_sweeps")
    ax1.set_ylim((mean_d2 - 100, mean_d2 + 100))
    ax0.set_ylim((-10, 210))

    ax0.scatter(sweeps, results_d1, label="Dicke erste Schicht", color="blue", marker="x", s=15)
    ax0.axhline(mean_d1, label=f"Durchschnittliche Dicke erste Schicht\n({mean_d1}$\pm${std_d1} $\mu$m)", c="blue",
                ls="dashed", lw=2.0)
    ax0.axhline(d1_truth, label=f"TSweeper Messung erste Schicht\n({d1_truth} $\mu$m)"
                , c="blue", lw=2.0)

    ax0.scatter(sweeps, results_d3, label="Dicke dritte Schicht", color="green", s=15)
    ax0.axhline(mean_d3, label=f"Durchschnittliche Dicke dritte Schicht\n({mean_d3}$\pm${std_d3} $\mu$m)", c="green",
                ls="dashed", lw=2.0)
    ax0.axhline(d3_truth, label=f"TSweeper Messung dritte Schicht\n({d3_truth} $\mu$m)",
                c="green", lw=2.0)

    ax1.scatter(sweeps, results_d2, label="Dricke zweite Schicht", c="red", s=15)
    ax1.axhline(mean_d2, label=f"Durchschnittliche Dicke zweite Schicht\n({mean_d2}$\pm${std_d2} $\mu$m)", c="red",
                ls="dashed", lw=2.0)
    ax1.axhline(d2_truth, label=f"TSweeper Messung zweite Schicht\n({d2_truth} $\mu$m)",
                c="red", lw=2.0)

    ax0.set_xlabel("Sweep Index")
    ax0.set_ylabel("Dicke bester Fit ($\mu$m)")
    ax1.set_xlabel("Sweep Index")
    ax1.set_ylabel("Dicke bester Fit ($\mu$m)")

    fig, (ax0, ax1) = plt.subplots(2, 1, num=str(sam_meas_.sample.name) + "_single_sweeps_losses")
    ax0.plot(sweeps, min_losses[0], label="Kleinstes Residuum erste Optimierung")
    ax0.plot(sweeps, min_losses[1], label="Kleinstes Residuum zweite Optimierung")

    ax1.plot(sweeps, results_d1, label="Dicke erste Schicht", c="blue")
    ax1.plot(sweeps, results_d2, label="Dicke zweite Schicht", c="red")
    ax1.plot(sweeps, results_d3, label="Dicke dritte Schicht", c="green")
    ax0.set_xlabel("Sweep Index")
    ax0.set_ylabel("Kleinstes Residuum")
    ax1.set_ylabel("Dicke bester Fit ($\mu$m)")
