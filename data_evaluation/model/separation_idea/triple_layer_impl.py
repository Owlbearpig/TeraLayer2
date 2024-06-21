import numpy as np
from tmm import list_snell, interface_r
from numpy import conj, abs, cos
from meas_eval.consts import thea, c_thz
from meas_eval.JumpingLaser.parse_data import Measurement, ModelMeasurement, SystemEnum, SamplesEnum
import matplotlib.pyplot as plt
from model.tmm_package import coh_tmm_slim
from functions import std_err
from scipy.optimize import minimize
from itertools import product
from pathlib import Path
import matplotlib as mpl
from helpers import read_opt_res_file

minimum_prec = 4


def plot_opt_res(res_):
    if not res_:
        return
    sam_meas = res_["sam_meas"]
    d1_truth_, d2_truth_, d3_truth_ = res_["d1_truth"], res_["d2_truth"], res_["d3_truth"]
    results_d1, results_d2, results_d3 = res_["results_d1"], res_["results_d2"], res_["results_d3"]
    mean_d1 = np.round(np.mean(results_d1), 2)
    mean_d2 = np.round(np.mean(results_d2), 2)
    mean_d3 = np.round(np.mean(results_d3), 2)
    std_d1 = np.round(std_err(results_d1), 2)
    std_d2 = np.round(std_err(results_d2), 2)
    std_d3 = np.round(std_err(results_d3), 2)

    d1_truth_ = np.round(d1_truth_, 2)
    d2_truth_ = np.round(d2_truth_, 2)
    d3_truth_ = np.round(d3_truth_, 2)

    sweeps = list(range(len(results_d1)))

    fig, (ax0, ax1) = plt.subplots(1, 2, num=str(sam_meas.sample.name) + "_single_sweeps")

    ax1.set_ylim((mean_d2 - 100, mean_d2 + 100))

    publication_plot = True
    if publication_plot:
        en_labels = False
        font_size = 26
        text_font = font_size - 6
        ax0.grid(False), ax1.grid(False)
        ax0.set_ylim((-15, 210))
    else:
        en_labels = True
        font_size = mpl.rcParams["font.size"]
        ax0.set_ylim((-10, 210))
        text_font = font_size

    x_pos = .5*len(sweeps)
    ax0.text(x_pos, 125, s="Layer 1", c="blue", size=text_font, ha="center")
    s = f"Mean (-): ({mean_d1}$\pm${std_d1}) $\mu$m\nNominal (--): {d1_truth_} $\mu$m"
    ax0.text(x_pos, 96, s=s, c="blue", size=text_font, ha="center")
    label = en_labels*"Dicke erste Schicht"
    ax0.scatter(sweeps, results_d1, label=label, color="blue", marker="o", s=10, alpha=0.15, linewidths=0)
    label = en_labels * f"Durchschnittliche Dicke erste Schicht\n({mean_d1}$\pm${std_d1} $\mu$m)"
    ax0.axhline(mean_d1, label=label, c="blue", lw=2, zorder=9)
    label = en_labels * f"TSweeper Messung erste Schicht\n({d1_truth_} $\mu$m)"
    ax0.axhline(d1_truth_, label=label, c="blue", ls="dashed", lw=2, zorder=9)

    ax1.text(x_pos, 710, s="Layer 2", c="red", size=text_font, ha="center")
    s = f"Mean (-): ({mean_d2}$\pm${std_d2}) $\mu$m\nNominal (--): {d2_truth_} $\mu$m"
    ax1.text(x_pos, 681, s=s, c="red", size=text_font, ha="center")
    label = en_labels * "Dricke zweite Schicht"
    ax1.scatter(sweeps, results_d2, label=label, c="red", s=10, alpha=0.15)
    label = en_labels * f"Durchschnittliche Dicke zweite Schicht\n({mean_d2}$\pm${std_d2} $\mu$m)"
    ax1.axhline(mean_d2, label=label, c="red", lw=2, zorder=9)
    label = en_labels * f"TSweeper Messung zweite Schicht\n({d2_truth_} $\mu$m)"
    ax1.axhline(d2_truth_, label=label, c="red", ls="dashed", lw=2, zorder=9)

    ax0.text(x_pos, 24, s="Layer 3", c="green", size=text_font, ha="center")
    s = f"Mean (-): ({mean_d3}$\pm${std_d3}) $\mu$m\nNominal (--): {d3_truth_} $\mu$m"
    ax0.text(x_pos, -5, s=s, c="green", size=text_font, ha="center")
    label = en_labels * "Dicke dritte Schicht"
    ax0.scatter(sweeps, results_d3, label=label, c="green", s=10, alpha=0.15)
    label = en_labels * f"Durchschnittliche Dicke dritte Schicht\n({mean_d3}$\pm${std_d3} $\mu$m)"
    ax0.axhline(mean_d3, label=label, c="green", lw=2, zorder=9)
    label = en_labels * f"TSweeper Messung dritte Schicht\n({d3_truth_} $\mu$m)"
    ax0.axhline(d3_truth_, label=label, c="green", ls="dashed", lw=2, zorder=9)

    if publication_plot:
        ax0.set_xlabel("Sweep number", size=font_size)
        ax0.set_ylabel("Layer thickness ($\mu$m)", size=font_size)
        ax1.set_xlabel("Sweep number", size=font_size)
        ax1.set_ylabel("Layer thickness ($\mu$m)", size=font_size)
        ax0.tick_params(axis='both', which='major', labelsize=font_size)
        ax0.tick_params(axis='both', which='minor', labelsize=font_size)
        ax1.tick_params(axis='both', which='major', labelsize=font_size)
        ax1.tick_params(axis='both', which='minor', labelsize=font_size)
    else:
        ax0.set_xlabel("Aufnahme Index", size=font_size)
        ax0.set_ylabel("Dicke bester Fit ($\mu$m)", size=font_size)
        ax1.set_xlabel("Aufnahme Index", size=font_size)
        ax1.set_ylabel("Dicke bester Fit ($\mu$m)", size=font_size)


def triple_layer_impl(sam_meas_: Measurement, ts_meas_: Measurement, options: dict):
    single_sweep_eval = options["single_sweep_eval"]
    selected_sweep_ = options["selected_sweep"]
    save_file = options["save_dir"] / f"opt_res_{str(sam_meas_.sample.name)}.txt"
    plot_grid = options["plot_grid"]

    d1_truth, d2_truth, d3_truth = sam_meas_.sample.value.thicknesses.astype(int)

    res = {"sam_meas": sam_meas_, "d1_truth": d1_truth, "d2_truth": d2_truth, "d3_truth": d3_truth}
    if options["read_res_if_exists"] and not single_sweep_eval:
        if save_file.exists():
            read_results = read_opt_res_file(save_file)
            if read_results:
                res.update(read_results)
                plot_opt_res(res)
                return
        else:
            pass

    save_file.touch(exist_ok=True)

    def grid_points(spacing=50, x_shift=0, y_shift=0):
        d11 = np.arange(d1[0] + x_shift, d1[-1], spacing)
        d22 = np.arange(d2[0] + y_shift, d2[-1], spacing)

        grid = list(product(d11, d22))

        return grid

    num_layers = 5  # first and last layers are air
    pol = "s"

    freqs = sam_meas_.freq

    is_ts_meas = sam_meas_.system.name == SystemEnum.TSweeper.name
    if is_ts_meas:
        freq_min, freq_max = 0.2, 1.25
        freq_min_idx, freq_max_idx = np.argmin(np.abs(freq_min - freqs)), np.argmin(np.abs(freq_max - freqs))
        f_res = 10
        freqs = sam_meas_.freq[freq_min_idx:freq_max_idx:f_res]

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
    d3 = np.arange(bounds[2][0], bounds[2][1] + 1, 1)
    if sam_meas_.sample == SamplesEnum.ampelMannRight:
        d3 = np.arange(bounds[2][0], 100, 1)

    if is_ts_meas:
        real_weights, imag_weights = np.ones((2, len(freqs)))
        weights = np.ones(len(freqs))
    else:
        real_weights, imag_weights = 1 / np.std(sam_meas_.r.real, axis=0), 1 / np.std(sam_meas_.r.imag, axis=0)
        real_weights, imag_weights = real_weights, imag_weights
        # real_weights, imag_weights = np.ones((2, len(freqs)))
        # real_weights = np.array([1, 1, 1, 1, 1, 0.0])
        # imag_weights = np.array([1, 1, 1, 1, 1, 0.0])
        weights = np.ones(len(freqs)) * (real_weights + imag_weights)
        if sam_meas_.system.value == SystemEnum.WaveSource.value and np.isclose(freqs[0], 0.05, atol=0.01):
            weights[0] = 0

    def eval_sample(sweep_idx=None, grid_=None):
        print(f"Evaluating sweep: {sweep_idx}, at {freqs} THz")
        if is_ts_meas:
            r_exp_ = sam_meas_.r_avg[freq_min_idx:freq_max_idx:f_res]
        else:
            if sweep_idx is None:
                r_exp_ = sam_meas_.r_avg
            else:
                r_exp_ = sam_meas_.r[sweep_idx]

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

        ###
        # extension for grid search ->
        ###
        if grid_:
            def expr1_sum_fun(p_):
                d1_, d2_ = p_
                loss = np.sum(np.array([expr1_(d1_, d2_, f_idx) for f_idx in range(len(freqs))]), axis=0)
                return np.log10(loss)

            def initial_simplex(x0_, spread=10):
                simplex = np.zeros((3, 2))
                for i in range(3):
                    for j in range(2):
                        if i - 1 == j:
                            simplex[i, j] = x0_[j] - spread
                        else:
                            simplex[i, j] = x0_[j]

                return simplex

            opt_bounds = [(d1[0], d1[-1]), (d2[0], d2[-1])]
            x0 = np.array([*grid_[0]], dtype=float)
            best_res = minimize(expr1_sum_fun, x0, bounds=opt_bounds, method="Nelder-Mead")
            best_start_val = None
            for grid_point_ in grid_:
                x0 = np.array([*grid_point_], dtype=float)
                res = minimize(expr1_sum_fun, x0, method="Nelder-Mead",
                               options={"initial_simplex": initial_simplex(x0)})

                if res.fun < best_res.fun:
                    best_res = res
                    best_start_val = x0
            x = np.round(best_res.x, 2)
            print(f"Grid opt minimum {np.round(best_res.fun, minimum_prec)} at ({x[0]}, {x[1]}) um")
            print(best_start_val)
            print(best_res)

            return best_start_val

        X, Y = np.meshgrid(d1, d2)

        expr1_err = np.array([expr1_(X, Y, f_idx) for f_idx in range(len(freqs))])
        expr1_err = (np.multiply(expr1_err.T, weights)).T
        expr1_sum_ = np.sum(expr1_err, axis=0)

        i, j = np.unravel_index(np.argmin(expr1_sum_), expr1_sum_.shape)
        d1_found_, d2_found_ = d1[j], d2[i]
        print(f"First OF minimum: {np.round(np.min(expr1_sum_), 3)}, at "
              f"({np.round(d1_found_, minimum_prec)}, {np.round(d2_found_, minimum_prec)}) um")

        # 2nd stage. Use p0 from previous 2D opt. problem to find last thickness
        def fun(p):
            if any(p) < 0:
                return np.inf

            r_exp_mod = np.zeros_like(freqs, dtype=complex)
            for freq_idx_ in range(len(freqs)):
                d = np.array([np.inf, *p, np.inf], dtype=float)
                r_exp_mod[freq_idx_] = -coh_tmm_slim(pol, n_list[freq_idx_], d, thea, lam_vac[freq_idx_])
            real_err = (r_exp_mod.real - r_exp_.real) ** 2
            imag_err = (r_exp_mod.imag - r_exp_.imag) ** 2

            tot_err = np.sum(real_err * real_weights + imag_err * imag_weights)

            return tot_err

        d3_found_, best_res_fun = None, np.inf
        if not is_ts_meas:
            opt_bounds = [(d1[0], d1[-1]), (d2[0], d2[-1]), (d3[0], d3[-1])]
            x0 = np.array([d1_found_, d2_found_, d3[len(d3) // 2]], dtype=float)
            best_res = minimize(fun, x0, bounds=opt_bounds, method="Nelder-Mead")
            for d3_ in [20.0, 70.0, 120.0, 170.0]:
                x0 = np.array([d1_found_, d2_found_, d3_], dtype=float)
                res = minimize(fun, x0, bounds=opt_bounds, method="Nelder-Mead")
                if res.fun < best_res.fun:
                    best_res = res
            x = np.round(best_res.x, 2)
            best_res_fun = best_res.fun
            print(f"2nd stage opt minimum {np.round(best_res_fun, minimum_prec)} at ({x[0]}, {x[1]}, {x[2]}) um")
            d1_found_, d2_found_, d3_found_ = x

        d3_err_ = []
        best_fit = (None, np.inf)
        for d3_ in d3:
            tot_err = fun(np.array([d1_found_, d2_found_, d3_], dtype=float))
            d3_err_.append(tot_err)
            if tot_err < best_fit[1]:
                best_fit = (d3_, tot_err)

        if not d3_found_:
            d3_found_ = d3[np.argmin(d3_err_)]

        print(f"Sweep opt. result: {np.round(np.min(expr1_sum_), minimum_prec)}, {np.round(best_fit[1], minimum_prec)}"
              f" at ({d1_found_}, {d2_found_}, {d3_found_}) um\n")

        return d1_found_, d2_found_, d3_found_, expr1_sum_, d3_err_

    sweep_s = "alle Aufnahmen" if selected_sweep_ is None else f"Aufnahme {selected_sweep_}"
    if is_ts_meas:
        sweep_s = ""

    d1_found, d2_found, d3_found, expr1_sum, d3_err = eval_sample(selected_sweep_)

    i, j = np.unravel_index(np.argmin(expr1_sum), expr1_sum.shape)
    expr1_min_d1, expr1_min_d2 = d1[j], d2[i]

    fig_num = (str(sam_meas_.sample.name) + f"_expr1_sum_sweep_{selected_sweep_}")
    s = "TSweeper" if is_ts_meas else sam_meas_.system.name
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
    s = f"({expr1_min_d1}, {expr1_min_d2}) {sam_meas_.system.name}"
    if is_ts_meas:
        s = s.replace(f"{sam_meas_.system.name}", "TSweeper")
    elif plot_grid:  # if True:
        grid = grid_points(spacing=50, x_shift=10, y_shift=10)
        for i, grid_point in enumerate(grid):
            if i == 0:
                ax.scatter(*grid_point, s=40, c="black", label="Anfangswertgitter")
            else:
                ax.scatter(*grid_point, s=40, c="black")
        best_x0 = eval_sample(grid_=grid)
        ax.scatter(*best_x0, s=40, c="red", label="Anfangswert der zum Optimum führt")

    ax.text(expr1_min_d1 + 5, expr1_min_d2 - 5, s, fontsize=18, c="red")
    ax.scatter(*(expr1_min_d1, expr1_min_d2), s=40, c="red", marker='x')
    ax.set_xlabel("Dicke erste Schicht ($\mu$m)")
    ax.set_ylabel("Dicke zweite Schicht ($\mu$m)")

    # plt.figure(str(sam_meas_.sample.name) + f"_avg_{sam_meas_.system.name}")
    plt.figure(str(sam_meas_.sample.name) + f"_sweep_{selected_sweep_}")
    plt.title(f"Residuum {sweep_s}")
    plt.plot(d3, np.log10(d3_err), label=f"Residuen {sam_meas_.system.name} {sweep_s}")
    min_point = (d3[np.argmin(d3_err)], np.log10(np.min(d3_err)))
    plt.annotate(f"{min_point[0]}, {np.round(min_point[1], minimum_prec)}", xy=(min_point[0], min_point[1]),
                 xytext=(-20, 20),
                 textcoords='offset points', ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                 color='red', mutation_scale=22))
    plt.xlabel("Dicke dritter Schicht ($\mu$m)")
    plt.ylabel("$\log_{10}$(Residuum)")

    if not single_sweep_eval and is_ts_meas:
        with open(save_file, "r") as file:
            lines = file.readlines()
            ts_str = f"TSweeper (um): {d1_found}, {d2_found}, {d3_found}\n"
            lines.insert(0, ts_str)

        with open(save_file, 'w') as file:
            file.writelines(lines)

    if single_sweep_eval or is_ts_meas:
        return

    n_sweeps = sam_meas_.n_sweeps
    sweeps = list(range(n_sweeps))
    results_d1, results_d2, results_d3 = np.zeros((3, n_sweeps), dtype=float)
    min_losses = np.zeros((2, n_sweeps), dtype=float)

    with open(save_file, "a") as file:
        file.write("sweep, d0 (um), d1 (um), d2 (um)\n")
        for sweep_idx in sweeps:
            d1_found, d2_found, d3_found, expr1_sum, d3_err = eval_sample(sweep_idx)

            results_d1[sweep_idx], results_d2[sweep_idx], results_d3[sweep_idx] = d1_found, d2_found, d3_found
            min_losses[:, sweep_idx] = np.min(expr1_sum), np.min(d3_err)
            file.write(f"{sweep_idx}, {d1_found}, {d2_found}, {d3_found}\n")

    mean_d1 = np.round(np.mean(results_d1), 2)
    mean_d2 = np.round(np.mean(results_d2), 2)
    mean_d3 = np.round(np.mean(results_d3), 2)
    std_d1 = np.round(std_err(results_d1), 2)
    std_d2 = np.round(std_err(results_d2), 2)
    std_d3 = np.round(std_err(results_d3), 2)

    with open(save_file, "r") as file:
        lines = file.readlines()
        mean_str = f"Mean: {mean_d1}±{std_d1}, {mean_d2}±{std_d2}, {mean_d3}±{std_d3}\n"
        lines.insert(1, mean_str)

    with open(save_file, 'w') as file:
        file.writelines(lines)

    res.update({"results_d1": results_d1, "results_d2": results_d2, "results_d3": results_d3})

    plot_opt_res(res)

    fig, (ax0, ax1) = plt.subplots(2, 1, num=str(sam_meas_.sample.name) + "_single_sweeps_losses")
    ax0.plot(sweeps, min_losses[0], label="Kleinstes Residuum erste Optimierung")
    ax0.plot(sweeps, min_losses[1], label="Kleinstes Residuum zweite Optimierung")

    ax1.plot(sweeps, results_d1, label="Dicke erste Schicht", c="blue")
    ax1.plot(sweeps, results_d2, label="Dicke zweite Schicht", c="red")
    ax1.plot(sweeps, results_d3, label="Dicke dritte Schicht", c="green")
    ax0.set_xlabel("Aufnahme Index")
    ax0.set_ylabel("Kleinstes Residuum")
    ax1.set_ylabel("Dicke bester Fit ($\mu$m)")
