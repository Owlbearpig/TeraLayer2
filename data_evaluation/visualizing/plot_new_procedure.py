from pathlib import Path
from consts import ROOT_DIR
import matplotlib.pyplot as plt
import numpy as np
from numpy import array

solutions = ROOT_DIR / "optimization" / "solutions_new.txt"


def fail_cnt(x, y):
    cnt = 0
    for i in range(len(x)):
        s = 0
        s += min(np.abs(x[i, 0]-y[i, 0]), np.abs(x[i, 0]-y[i, 2]))
        s += np.abs(x[i, 1]-y[i, 1])
        s += min(np.abs(x[i, 2] - y[i, 2]), np.abs(x[i, 2] - y[i, 0]))
        if s > 10:
            cnt += 1

    return cnt


results, truths, fevals = [], [], []
with open(solutions, "r") as file:
    for line_idx, line in enumerate(file.readlines()):
        #if (line_idx >= 731)*(line_idx <= 830): # bh algo
        #if (line_idx >= 838) * (line_idx <= 936): # shgo
        if (1365 <= line_idx) * (line_idx <= 1464): # grid search, real data
        #if (1260 <= line_idx) * (line_idx <= 1359): # shgo, real data
            split_line = line.split(" __ ")
            fevals.append(float(split_line[5]))
            truth, res = eval(split_line[0]), eval(split_line[1])
            #truth = [42.5, 641.3, 74.4]
            results.append(res)
            truths.append(truth)



print(len(results))

print(f"Fevals average: {np.mean(fevals)}")
results, truths = array(results), array(truths)
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
ax2.scatter(sample_idx_py, results[:, 0], label=f"d{0} opt. res.", s=dot_size, zorder=2, color=color_lst[0], marker=(5, 2))
ax.scatter(sample_idx_py, results[:, 1], label=f"d{1} opt. res.", s=dot_size, zorder=2, color=color_lst[1], marker=(5, 2))
ax2.scatter(0, -10, label=f"d{1} opt. res.", s=dot_size, zorder=2, color=color_lst[1], marker=(5, 2)) # legend hack ...
ax2.scatter(sample_idx_py, results[:, 2], label=f"d{2} opt. res.", s=dot_size, zorder=2, color=color_lst[2], marker=(5, 2))

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

ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
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
#plt.title(f"Fail count: {fail_cnt(results, truths)} (summed diff. > 15)")
plt.title(f"Mean values: $d_0$: {np.round(np.mean(results[:, 0]))}$\pm${np.round(np.std(results[:, 0]))}, "
          f"$d_1$: {np.round(np.mean(results[:, 1]))}$\pm${np.round(np.std(results[:, 1]))}, "
          f"$d_2$: {np.round(np.mean(results[:, 2]))}$\pm${np.round(np.std(results[:, 2]))}")
plt.xlabel("Measurement idx")
#plt.legend(loc=(0.8, 0.7))
plt.legend(loc=(0.85, 0.95))
plt.show()
