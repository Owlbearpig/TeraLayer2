import matplotlib.pyplot as plt
from twos_compl_OF_v2 import CostFuncFixedPoint
from functools import partial
from optimization.nelder_mead_nD import nm_gridsearch
from twos_compl_datatype import Bin2sComp
from numfi import numfi as numfi_
from numpy import array, pi
from consts import selected_freqs
import numpy as np
import time


if __name__ == '__main__':
    np.random.seed(420)
    p_sol = array([241., 661., 237.])
    noise_factor = 0.00
    sam_idx = None

    pd, p = 4, 15
    cost_inst = CostFuncFixedPoint(pd=pd, p=p, p_sol=p_sol, noise=noise_factor, plt_mod=True, sam_idx=sam_idx)
    cost_func = cost_inst.cost

    numfi = partial(numfi_, s=1, w=pd + p, f=p, fixed=True, rounding='floor')

    p0 = array([150, 600, 150])  # shouldn't change
    grid_spacing, size = 40, 3

    options = {"grid_spacing": grid_spacing, "iterations": 15, "numfi": numfi,
               "size": size, "verbose": False, "enhance_step": False, "simplex_spread": 40, "input_scale": 6}
    start = time.process_time()

    res = nm_gridsearch(cost_func, p0, options)

    r_mod = cost_inst.cost(res["x_downscaled"], ret_mod=True)
    r_sol = cost_inst.cost(array([45.77, 660.0, 72.6]) / res["upscale"], ret_mod=True)

    if sam_idx == -1:
        print("Truth: ", p_sol)

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
