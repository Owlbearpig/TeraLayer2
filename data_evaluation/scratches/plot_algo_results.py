import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

from consts import ROOT_DIR

sim_result_path = ROOT_DIR / Path("scratches") / Path("convertedsim_output.txt")
python_result_path = ROOT_DIR / Path("optimization") / Path("solutions.txt")

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

plt.plot(python_result[:, 0], label="d0 python")
plt.plot(sim_results[:, 0], ".", label="d0 fpga sim")

plt.plot(python_result[:, 1], label="d1 python", color="b")
plt.plot(sim_results[:, 1], ".", label="d1 fpga sim")

plt.plot(python_result[:, 2], label="d2 python")
plt.plot(sim_results[:, 2], ".", label="d2 fpga sim")

plt.ylabel("Layer width (µm)")
plt.xlabel("Sample idx")
plt.legend()
plt.show()


