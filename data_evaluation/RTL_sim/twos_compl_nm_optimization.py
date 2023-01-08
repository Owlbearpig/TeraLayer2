from twos_compl_OF import cost
from functools import partial
from optimization.nelder_mead_nD import nm_gridsearch
from twos_compl_datatype import Bin2sComp
import numpy as np
import time


if __name__ == '__main__':
    np.random.seed(420)
    p_sol = [290.0, 658.0, 94.0]
    pd, p = 12, 25

    cost_func = partial(cost, pd=pd, p=p)

    p0 = [150, 600, 150]  # shouldn't change
    p0 = [Bin2sComp(x, pd = pd, p = p) for x in p0]
    grid_spacing, size = 50, 3

    options = {"grid_spacing": grid_spacing, "iterations": 15,
               "size": size, "verbose": False, "enhance_step": False}
    start = time.process_time()
    res = nm_gridsearch(cost_func, p0, options)

    print("Truth: ", p_sol)
    print("Found: ", res["x"], res["fun"])
    total_runtime = (750 / 3780) * res["total_iters"]
    print("Total iterations:", res["total_iters"], f"estimated FPGA runtime: {total_runtime} us")
    print("Runtime: ", time.process_time() - start, "(s)")
    print("Total obj. func. calls: ", res["nfev"])
