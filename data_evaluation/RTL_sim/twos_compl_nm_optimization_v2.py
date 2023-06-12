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


def single_measurement(sam_idx, en_plt=False, debug=False):
    np.random.seed(420)
    p_sol = array([241., 661., 237.])
    noise_factor = 0.00

    pd, p = 4, 11
    # pd, p = 4, 21
    cost_inst = CostFuncFixedPoint(pd=pd, p=p, p_sol=p_sol, noise=noise_factor, en_plt=en_plt, sam_idx=sam_idx)
    cost_func = cost_inst.cost

    numfi = partial(numfi_, s=1, w=pd + p, f=p, fixed=True, rounding='floor')

    p0 = array([150, 600, 150])  # shouldn't change
    grid_spacing = 40
    size = 3
    simplex_spread = 40
    iterations = 15
    input_scale = 6  # don't change

    options = {"grid_spacing": grid_spacing, "iterations": iterations, "numfi": numfi, "size": size,
               "verbose": False, "enhance_step": False, "simplex_spread": simplex_spread, "input_scale": input_scale,
               "debug": debug}

    start = time.process_time()

    res = nm_gridsearch(cost_func, p0, options)

    r_mod = cost_inst.cost(res["x_downscaled"], ret_mod=True)
    r_sol = cost_inst.cost(array([45.77, 660.0, 72.6]) / res["upscale"], ret_mod=True)

    if sam_idx == -1:
        print("Truth: ", p_sol)

    print("Found: ", res["x"], res["fun"])
    est_fpga_runtime = (750 / 3780) * res["total_iters"]
    print("Total iterations:", res["total_iters"], f"Estimated FPGA runtime: {est_fpga_runtime} us")
    print("Runtime: ", time.process_time() - start, "(s)")
    print("Total obj. func. calls: ", res["nfev"])

    if en_plt:
        plt.figure("Measurement")
        plt.title(f"Found: " + str(res["x"]))
        plt.plot(selected_freqs, np.abs(r_mod), label=f"model amplitude {sam_idx}")
        plt.plot(selected_freqs, np.angle(r_mod), label=f"model phase {sam_idx}")
        plt.plot(selected_freqs, np.abs(r_sol), label=f"solution amplitude {sam_idx}")
        plt.plot(selected_freqs, np.angle(r_sol), label=f"solution phase {sam_idx}")
        plt.legend()
        plt.show()

    return res["x"]


def all_measurements():
    d0, d1, d2 = [], [], []
    for idx in range(101):
        res = single_measurement(idx)
        d0.append(res["x"][0])
        d1.append(res["x"][1])
        d2.append(res["x"][2])

    plt.plot(d0)
    plt.plot(d1)
    plt.plot(d2)
    plt.show()


def main():
    single_measurement(sam_idx=1, debug=True)
    # all_measurements()


if __name__ == '__main__':
    main()
