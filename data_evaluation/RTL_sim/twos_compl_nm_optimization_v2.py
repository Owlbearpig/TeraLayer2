from twos_compl_OF_v2 import CostFuncFixedPoint
from functools import partial
from optimization.nelder_mead_nD import nm_gridsearch
from twos_compl_datatype import Bin2sComp
from numfi import numfi as numfi_
from numpy import array, pi
import numpy as np
import time


if __name__ == '__main__':
    np.random.seed(420)
    p_sol = [282.0, 509.0, 50.0]

    pd, p = 4, 23
    cost_func = CostFuncFixedPoint(pd=pd, p=p, p_sol=p_sol).cost
    numfi = partial(numfi_, s=1, w=pd + p, f=p, fixed=True, rounding='floor')

    p0 = array([150, 600, 150])  # shouldn't change
    grid_spacing, size = 50, 3

    options = {"grid_spacing": grid_spacing, "iterations": 15, "numfi": numfi,
               "size": size, "verbose": False, "enhance_step": False, "simplex_spread": 40}
    start = time.process_time()

    res = nm_gridsearch(cost_func, p0, options)

    print("Truth: ", p_sol)
    print("Found: ", res["x"], res["fun"])
    total_runtime = (750 / 3780) * res["total_iters"]
    print("Total iterations:", res["total_iters"], f"Estimated FPGA runtime: {total_runtime} us")
    print("Runtime: ", time.process_time() - start, "(s)")
    print("Total obj. func. calls: ", res["nfev"])
