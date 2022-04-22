import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

from consts import ROOT_DIR

# converted_sim_output_13bit_prec p=22
# converted_sim_output_13bit_prec2 p=17
# converted_sim_output_13bit_prec3 p=16
# converted_sim_output_13bit_prec4 p=15
# converted_sim_output_13bit_prec5 p=14
# converted_sim_output_13bit_prec6 p=13
# converted_sim_output_13bit_prec7 p=12
# converted_sim_output_13bit_prec8 p=8
# converted_sim_output_13bit_prec9 p=10
sim_result_path = ROOT_DIR / "scratches" / "vivado_sim_results" / "converted_sim_output_13bit_prec.txt"
python_result_path = ROOT_DIR / "optimization" / "solutions_oldversion.txt"

measurement_limit = 100

sim_results = []
with open(sim_result_path) as file:
    for idx, line in enumerate(file.readlines()):
        if idx == measurement_limit:
            break
        res = ""
        res += line.split("[")[1]
        res = res.split("]")[0]
        sim_results.append(list(eval(res)))

python_result = []
with open(python_result_path) as file:
    for idx, line in enumerate(file.readlines()):
        if idx == measurement_limit:
            break
        line = list(eval(line)[0])
        python_result.append(line)

python_result = np.array(python_result)
sim_results = np.array(sim_results)

sample_idx_py, sample_idx_fpga = range(len(python_result[:, 0])), range(len(sim_results[:, 0]))

f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

line_width, dot_size = 2.5, 35
plt.rcParams.update({'font.size': 16})

color_lst = ["black", "red", "black"]

py_only = True

for i in range(3):
    if py_only:
        continue
    ax.scatter(sample_idx_fpga, sim_results[:, i], label=f"d{i} vivado sim", s=dot_size, zorder=2, color=color_lst[i])
    ax2.scatter(sample_idx_fpga, sim_results[:, i], label=f"d{i} vivado sim", s=dot_size, zorder=2, color=color_lst[i])
for i in range(3):
    ax.plot(sample_idx_py, python_result[:, i], label=f"d{i} python", linewidth=line_width, zorder=1)
    ax2.plot(sample_idx_py, python_result[:, i], label=f"d{i} python", linewidth=line_width, zorder=1)

ax.set_ylim(600, 650)  # outliers only
ax2.set_ylim(0, 100)  # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = 0.01  # how big to make the diagonal lines in axes coordinates

# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)

# ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

if not py_only:
    ax.text(10, 620, "vivado simulation means:")
for i in range(3):
    if py_only:
        continue
    s = f"$mean(d_{i}) = {np.round(np.mean(sim_results[:, i]), 2)}\pm{np.round(np.std(sim_results[:, i]), 2)}$ $\mu m$"
    ax.text(10, 617 - i * 3, s, )

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
plt.xlabel("Measurement idx")
plt.legend(loc=(0.8, 0.7))
plt.show()
