from pathlib import Path
from consts import ROOT_DIR
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from scratches.snippets.base_converters import twos_comp_list_to_dec

fail_threshold = 15


def deviation(x, y):
    devs = []
    for i in range(len(x)):
        dev_d0 = np.abs(x[i, 0] - y[i, 0])
        dev_d1 = np.abs(x[i, 1] - y[i, 1])
        dev_d2 = np.abs(x[i, 2] - y[i, 2])
        devs.append([dev_d0, dev_d1, dev_d2])

    return array(devs)


def fail_cnt(x, y):
    devs = deviation(x, y)

    cnt, max_devs = 0, []
    for i in range(len(x)):
        if any(devs[i] >= fail_threshold):
            cnt += 1
            max_dev = round(max(devs[i]), 1)
            print(f"Failed, result: {x[i]}, target: {y[i]}. Max dev: {max_dev}")
            max_devs.append(max_dev)

    print(f"Avg. max. dev. of fails: {np.round(np.mean(max_devs), 1)}pm{np.round(np.std(max_devs), 1)}", )
    return cnt


solutions = ROOT_DIR / "optimization" / "results_nm_grid.txt"


# vivado_sim_output_path = r"H:\IPs\eval\proj_eval1\proj_eval1.sim\sim_1\behav\xsim\sim_output.txt"
#

def read_optim_result(file_path):
    # sam_idx __ found __ fx __ p0 __ fevals __ opt_p0
    sam_idx_lst, found_lst, fevals_lst = [], [], []
    with open(file_path, "r") as file:
        for line_idx, line in enumerate(file.readlines()):
            idx_start = 0
            line_idx += 1
            if (line_idx >= idx_start) * (line_idx <= idx_start + 110):
                split_line = line.split(" __ ")
                try:
                    fevals_lst.append(float(split_line[4]))
                    sam_idx, found = eval(split_line[0]), eval(split_line[1])
                except Exception:
                    continue
                sam_idx_lst.append(sam_idx)
                found_lst.append(found)

    return sam_idx_lst, array(found_lst), fevals_lst


def read_result_file(file_path=solutions, vivado_sim=False, p=22):
    # pm0.25, 0.20, 0.15, 0.10, 0.05, 0.00
    # noise: 321, 427, 533, 639, 745, 851
    results, truths, fevals = [], [], []
    with open(file_path, "r") as file:
        for line_idx, line in enumerate(file.readlines()):
            idx_start = 109
            line_idx += 1
            if (line_idx >= idx_start) * (line_idx <= idx_start + 100):
                split_line = line.split(" __ ")
                try:
                    if not vivado_sim:
                        fevals.append(float(split_line[5]))
                except Exception:
                    continue
                if vivado_sim:
                    truth, res = eval(split_line[0]), twos_comp_list_to_dec(split_line[1], p)
                else:
                    truth, res = eval(split_line[0]), eval(split_line[1])
                results.append(res)
                truths.append(truth)

    return results, truths, fevals



plot_version_2 = False
plot_optim_res_realdata = True
if plot_version_2:
    results, truths, fevals = read_result_file(solutions) #MODEL SIM
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
elif plot_optim_res_realdata:
    res_file_path = ROOT_DIR / "optimization" / "FP_results_realdata_p11_v1.txt"
    sam_idx_lst, results, fevals = read_optim_result(res_file_path)

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    sample_idx_py = range(len(results))
    line_width, dot_size = 2.5, 45
    plt.rcParams.update({'font.size': 16})

    color_lst = ["blue", "red", "green"]

    #ax2.scatter(sample_idx_py, truths[:, 0], label=f"d{0} truth", s=dot_size, zorder=1, color=color_lst[0], alpha=0.24)
    #ax.scatter(sample_idx_py, truths[:, 1], label=f"d{1} truth", s=dot_size, zorder=1, color=color_lst[1], alpha=0.24)
    #ax2.scatter(0, -10, label=f"d{1} truth", s=dot_size, zorder=1, color=color_lst[1], alpha=0.24)  # legend hack ...
    #ax2.scatter(sample_idx_py, truths[:, 2], label=f"d{2} truth", s=dot_size, zorder=1, color=color_lst[2], alpha=0.24)

    dot_size -= 20
    ax2.scatter(sample_idx_py, results[:, 0], label=f"d{0} opt. res.", s=dot_size, zorder=2, color=color_lst[0],
                marker=(5, 2))
    ax.scatter(sample_idx_py, results[:, 1], label=f"d{1} opt. res.", s=dot_size, zorder=2, color=color_lst[1],
               marker=(5, 2))
    ax2.scatter(0, -10, label=f"d{1} opt. res.", s=dot_size, zorder=2, color=color_lst[1],
                marker=(5, 2))  # legend hack ...
    ax2.scatter(sample_idx_py, results[:, 2], label=f"d{2} opt. res.", s=dot_size, zorder=2, color=color_lst[2],
                marker=(5, 2))

    ax.set_ylim(580, 720)  # outliers only
    ax2.set_ylim(-10, 110)  # most of the data

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = 0.01  # how big to make the diagonal lines in axes coordinates

    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)

    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax.xaxis.set_tick_params(width=7)
    ax.yaxis.set_tick_params(width=7)
    ax.tick_params(direction='in', length=10, width=3, labelsize=18)
    ax.yaxis.label.set_size(24)
    ax.xaxis.label.set_size(24)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)

    ax2.xaxis.set_tick_params(width=7)
    ax2.yaxis.set_tick_params(width=7)
    ax2.tick_params(direction='in', length=10, width=3, labelsize=18)
    ax2.yaxis.label.set_size(24)
    ax2.xaxis.label.set_size(24)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(3)

    ax.set_ylabel("Layer width (µm)")
    ax.yaxis.set_label_coords(-0.05, -0.0)
    ax.set_title(f"Avg. result: {np.round(np.mean(array(results), axis=0), 1)} (µm), (10 \"decimal places\")")
    std_dev = np.std(array(results), axis=0)

    plt.title(f"Avg. deviation: "
              f"$d_0$: {np.round((std_dev[0]), 1)} (µm), "
              f"$d_1$: {np.round((std_dev[1]), 1)} (µm), "
              f"$d_2$: {np.round((std_dev[2]), 1)} (µm)")
    plt.xlabel("Measurement index")
    # plt.legend(loc=(0.8, 0.7))
    plt.legend(loc=(0.85, 0.95))
    plt.show()

else:
    vivado_sim_output_path = r"H:\IPs\eval\proj_eval1\proj_eval1.sim\sim_1\behav\xsim\sim_output_noise.txt"

    results, truths, fevals = read_result_file(vivado_sim_output_path, vivado_sim=True)
    sample_idx_py = range(len(results))
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    line_width, dot_size = 2.5, 45
    plt.rcParams.update({'font.size': 16})

    color_lst = ["blue", "red", "green"]

    ax2.scatter(sample_idx_py, truths[:, 0], label=f"d{0} truth", s=dot_size, zorder=1, color=color_lst[0], alpha=0.24)
    ax.scatter(sample_idx_py, truths[:, 1], label=f"d{1} truth", s=dot_size, zorder=1, color=color_lst[1], alpha=0.24)
    ax2.scatter(0, -10, label=f"d{1} truth", s=dot_size, zorder=1, color=color_lst[1], alpha=0.24)  # legend hack ...
    ax2.scatter(sample_idx_py, truths[:, 2], label=f"d{2} truth", s=dot_size, zorder=1, color=color_lst[2], alpha=0.24)

    dot_size -= 20
    ax2.scatter(sample_idx_py, results[:, 0], label=f"d{0} opt. res.", s=dot_size, zorder=2, color=color_lst[0],
                marker=(5, 2))
    ax.scatter(sample_idx_py, results[:, 1], label=f"d{1} opt. res.", s=dot_size, zorder=2, color=color_lst[1],
               marker=(5, 2))
    ax2.scatter(0, -10, label=f"d{1} opt. res.", s=dot_size, zorder=2, color=color_lst[1],
                marker=(5, 2))  # legend hack ...
    ax2.scatter(sample_idx_py, results[:, 2], label=f"d{2} opt. res.", s=dot_size, zorder=2, color=color_lst[2],
                marker=(5, 2))

    ax.set_ylim(495, 705)  # outliers only
    ax2.set_ylim(0, 325)  # most of the data

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = 0.01  # how big to make the diagonal lines in axes coordinates

    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)

    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax.xaxis.set_tick_params(width=7)
    ax.yaxis.set_tick_params(width=7)
    ax.tick_params(direction='in', length=10, width=3, labelsize=18)
    ax.yaxis.label.set_size(24)
    ax.xaxis.label.set_size(24)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)

    ax2.xaxis.set_tick_params(width=7)
    ax2.yaxis.set_tick_params(width=7)
    ax2.tick_params(direction='in', length=10, width=3, labelsize=18)
    ax2.yaxis.label.set_size(24)
    ax2.xaxis.label.set_size(24)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(3)

    ax.set_ylabel("Layer width (µm)")
    ax.yaxis.set_label_coords(-0.05, -0.0)
    ax.set_title(f"Fail count: {fail_cnt(results, truths)} (Max diff. > {fail_threshold} (µm))")

    dev = deviation(results, truths)
    plt.title(f"Avg. deviation: "
              f"$d_0$: {np.round(np.mean(dev[:, 0]), 1)} (µm), "
              f"$d_1$: {np.round(np.mean(dev[:, 1]), 1)} (µm), "
              f"$d_2$: {np.round(np.mean(dev[:, 2]), 1)} (µm)")
    plt.xlabel("Measurement")
    # plt.legend(loc=(0.8, 0.7))
    plt.legend(loc=(0.85, 0.95))
    plt.show()
