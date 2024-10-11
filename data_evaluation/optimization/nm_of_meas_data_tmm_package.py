import matplotlib.pyplot as plt
from model.of_meas_data_tmm_package import CostTMM
from functools import partial
from optimization.nelder_mead_nD import nm_gridsearch

from numfi import numfi as numfi_
from numpy import array, pi
from consts import selected_freqs
import numpy as np
import time


if __name__ == '__main__':
    sam_idx = 45  # np.random.randint(0, 101)

    cost_inst = CostTMM(sam_idx)
    cost_func = cost_inst.cost

    p0 = array([150, 600, 150])  # shouldn't change
    grid_spacing, size = 40, 3

    options = {"grid_spacing": grid_spacing, "iterations": 15, "size": size,
               "verbose": False, "enhance_step": False, "simplex_spread": 40, "input_scale": 6}

    start = time.process_time()

    res = nm_gridsearch(cost_func, p0, options)

    r_mod = cost_inst.cost(res["x"], ret_mod=True)
    r_sol = cost_inst.cost(array([45.77, 660.0, 72.6]), ret_mod=True)

    print("Found: ", res["x"], res["fun"])
    total_runtime = (750 / 3780) * res["total_iters"]
    print("Total iterations:", res["total_iters"], f"Estimated FPGA runtime: {total_runtime} us")
    print("Runtime: ", time.process_time() - start, "(s)")
    print("Total obj. func. calls: ", res["nfev"])

    plt.figure("Measurement")
    plt.title(f"Found: " + str(res["x"]))
    plt.plot(selected_freqs, np.abs(r_mod), label=f"model amplitude {sam_idx}")
    plt.plot(selected_freqs, np.angle(r_mod), label=f"model phase {sam_idx}")
    plt.plot(selected_freqs, np.abs(r_sol), label=f"solution amplitude {sam_idx}")
    plt.plot(selected_freqs, np.angle(r_sol), label=f"solution phase {sam_idx}")
    plt.legend()
    plt.show()
