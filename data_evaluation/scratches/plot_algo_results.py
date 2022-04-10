import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

from consts import ROOT_DIR
# converted_sim_output_13bit_prec p=22
# converted_sim_output_13bit_prec2 p=17
sim_result_path = ROOT_DIR / "scratches" / "vivado_sim_results" / "converted_sim_output_13bit_prec2.txt"
python_result_path = ROOT_DIR / "optimization" / "solutions_workingversion.txt"

sim_results = []
with open(sim_result_path) as file:
    for line in file.readlines():
        res = ""
        res += line.split("[")[1]
        res = res.split("]")[0]
        sim_results.append(list(eval(res)))

python_result = []
with open(python_result_path) as file:
    for line in file.readlines():
        line = list(eval(line)[0])
        python_result.append(line)

python_result = np.array(python_result)
sim_results = np.array(sim_results)

sample_idx_py, sample_idx_fpga = range(len(python_result[:, 0])), range(len(sim_results[:, 0]))

plt.plot(sample_idx_py, python_result[:, 0], label="d0 python")
plt.plot(sample_idx_py, python_result[:, 1], label="d1 python", color="b")
plt.plot(sample_idx_py, python_result[:, 2], label="d2 python")

plt.plot(sample_idx_fpga, sim_results[:, 0], ".", label="d0 vivado sim")
plt.plot(sample_idx_fpga, sim_results[:, 1], ".", label="d1 vivado sim")
plt.plot(sample_idx_fpga, sim_results[:, 2], ".", label="d2 vivado sim")

plt.text(10, 530, "vivado sim means:")
for i in range(3):
    s = f"$mean(d_{i}) = {np.round(np.mean(sim_results[:, i]), 2)}\pm{np.round(np.std(sim_results[:, i]), 2)}$ $\mu m$"
    plt.text(10, 500-i*30, s)

plt.ylabel("Layer width (µm)")
plt.xlabel("Sample idx")
plt.legend()
plt.show()


