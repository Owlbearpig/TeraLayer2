import numpy as np
from numpy import array
from triple_layer import r_fn, kz_list, d1, d2, d3, minimize, n_list, thea, lam, freqs, pol, coeffs, meas_sim
from model.tmm_package import coh_tmm_slim
import matplotlib.pyplot as plt


def xy_algo(r_exp):
    c0, c1, c2, c3, c4, c5, c6, c7 = coeffs(r_exp)

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

        s = np.abs(num) - np.abs(den)

        return s ** 2

    def fun11(x):
        return (expr1_(*x, 0) + expr1_(*x, 1) + expr1_(*x, 2) + expr1_(*x, 3) +
                expr1_(*x, 4) + expr1_(*x, 5))

    sum_expr1xy_1D_0 = np.sum([expr1xy_(0, d2, i) for i in range(6)], axis=0)
    sum_expr1xy_1D_500 = np.sum([expr1xy_(500, d2, i) for i in range(6)], axis=0)
    # sum_expr1xy_1D_0 = expr1xy_(0, d2, 0)
    # sum_expr1xy_1D_500 = expr1xy_(500, d2, 0)
    a = -0.5429

    if np.min(sum_expr1xy_1D_0) < np.min(sum_expr1xy_1D_500):
        # print(np.min(sum_expr1xy_1D_0))
        shift = d2[np.argmin(sum_expr1xy_1D_0)]
        b = shift
    else:
        # print(np.min(sum_expr1xy_1D_500))
        shift = d2[np.argmin(sum_expr1xy_1D_500)]
        b = shift - a * d2[-1]
    # print(shift, b)

    diag_line = []
    for d1_ in range(0, d1[-1], 5):
        d2_ = d1_ * a + b
        if 0 < d2_ < d2[-1]:
            diag_line.append((d1_, d2_))

    # print(diag_line)
    # diag_line = list(zip(np.arange(0, d1[-1], 5), b + a*np.arange(0, d1[-1], 5)))
    # print(diag_line)

    tot_nfev = 2 * len(sum_expr1xy_1D_0)
    grid = []
    for i in range(-15, 20, 5):
        grid.extend([(pt[0], pt[1] - i) for pt in diag_line])

    all_points = sorted([(pt, fun11(pt)) for pt in grid], key=lambda x: x[1])
    tot_nfev += len(all_points)
    # print(all_points[:5])
    best_point = all_points[0]

    opt_res = minimize(fun11, x0=best_point[0])

    opt_res["nfev"] += tot_nfev

    r_exp_mod = np.zeros_like(freqs, dtype=complex)
    y_vals = []
    best_fit = (-1, np.inf)
    for d3_ in d3:
        for freq_idx_ in range(len(freqs)):
            d = array([np.inf, *opt_res["x"], d3_, np.inf], dtype=float)
            r_exp_mod[freq_idx_] = -coh_tmm_slim(pol, n_list[freq_idx_], d, thea, lam[freq_idx_])
        err = np.sum((r_exp_mod.real - r_exp.real) ** 2 + (r_exp_mod.imag - r_exp.imag) ** 2)
        y_vals.append(err)
        if err < best_fit[1]:
            best_fit = (d3_, err)

    opt_res["nfev"] += len(d3)
    opt_res["x"] = np.array([*opt_res["x"], best_fit[0]])

    return opt_res


if __name__ == '__main__':
    from triple_layer import seed
    from functions import gen_p_sols
    from visualizing.plot_new_procedure import deviation, fail_cnt, fail_threshold
    bounds = [(1, 500), (1, 500), (1, 500)]
    cnt = 100
    results_list, truth_list, fevals_list = [], [], []
    test_values = gen_p_sols(cnt=cnt, seed=seed, layer_cnt=3, bounds=bounds)
    for d_truth in test_values:
        r_exp = meas_sim(d_truth)
        res = xy_algo(r_exp)

        print(res["x"])
        print(d_truth, "\n")

        results_list.append(res["x"])
        truth_list.append(d_truth)
        fevals_list.append(res["nfev"])

    results, truths, fevals = results_list, truth_list, fevals_list
    print(f"Identified {len(results)} / 100 entries")

    print(f"Fevals average: {np.mean(fevals)}")
    results, truths = array(results), array(truths)
    sample_idx_py = range(len(results))

    plt.rcParams['figure.constrained_layout.use'] = True
    f, axes = plt.subplots(3, 1, sharex=True)
    ax0, ax1, ax2 = axes
    line_width, dot_size = 2.5, 45
    plt.rcParams.update({'font.size': 16})

    color_lst = ["blue", "red", "green"]

    ax0.plot(sample_idx_py, truths[:, 0], label=f"$d_{0}$ truth", color=color_lst[0], alpha=0.4)
    ax1.plot(sample_idx_py, truths[:, 1], label=f"$d_{1}$ truth", color=color_lst[1], alpha=0.4)
    ax2.plot(sample_idx_py, truths[:, 2], label=f"$d_{2}$ truth", color=color_lst[2], alpha=0.4)

    dot_size -= 20
    ax0.scatter(sample_idx_py, results[:, 0], label=f"$d_{0}$ opt. res.", s=dot_size, zorder=2, color=color_lst[0],
                marker=(5, 2))
    ax1.scatter(sample_idx_py, results[:, 1], label=f"$d_{1}$ opt. res.", s=dot_size, zorder=2, color=color_lst[1],
                marker=(5, 2))

    ax2.scatter(sample_idx_py, results[:, 2], label=f"$d_{2}$ opt. res.", s=dot_size, zorder=2, color=color_lst[2],
                marker=(5, 2))

    ax0.set_title(f"Fail count: {fail_cnt(results, truths)} (Max diff. > {fail_threshold} (µm))")

    ax0.set_ylim(0, 350)
    ax1.set_ylim(495, 705)
    ax2.set_ylim(0, 350)

    for i, ax in enumerate(axes):
        ax.legend()
        ax.set_ylabel(f"$d_{i}$ Layer width (µm)")

    dev = deviation(results, truths)
    plt.title(f"Avg. deviation: "
              f"$d_0$: {np.round(np.mean(dev[:, 0]), 1)} (µm), "
              f"$d_1$: {np.round(np.mean(dev[:, 1]), 1)} (µm), "
              f"$d_2$: {np.round(np.mean(dev[:, 2]), 1)} (µm)")
    plt.xlabel("Measurement")

    plt.show()
