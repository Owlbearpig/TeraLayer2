import matplotlib.pyplot as plt
from consts import *
import numpy as np
from numpy import array, sum
import matplotlib as mpl
from model.cost_function import Cost
from functions import gen_p_sols
from functools import partial
from numfi import numfi as numfi_
from scipy.optimize import basinhopping, shgo
from optimization.nelder_mead_nD import nm_gridsearch
from RTL_sim.twos_compl_OF_v2 import CostFuncFixedPoint

# mpl.rcParams['lines.linestyle'] = '--'
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['xtick.major.width'] = 2.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
# plt.style.use(['dark_background'])
# plt.xkcd()
mpl.rcParams.update({'font.size': 22})


if __name__ == '__main__':
    ## grid options
    grid_spacing = 40
    size = 3
    # nm options
    simplex_spread = 40
    iterations = 15
    # noise options
    noise_factor = 0.0

    pd, p = 4, 15
    dir_ = Path("results") / Path(f"FP_pd{pd}_p{p}_cw")
    dir_.mkdir(exist_ok=True)
    numfi = partial(numfi_, s=1, w=pd + p, f=p, fixed=True, rounding='floor')

    test_values = np.arange(0, 101, 1)
    sols, fevals_all = [], []
    with open(dir_ / f"FP_results_nm_grid_cw_v1.txt", "a") as file:
        description = f"FP_p0_Gridsearch cw, "
        description += f"Iters={iterations}, size={size}, grid_spacing={grid_spacing}, pd={pd}, p={p}"
        description += f", simplex_spread={simplex_spread}"
        header = description + "\nsam_idx __ found __ fx __ p0 __ fevals __ opt_p0"
        file.write(header + "\n")

        for sam_idx in test_values:
            cost_func = CostFuncFixedPoint(pd=pd, p=p, sam_idx=sam_idx).cost

            p0 = array([150, 600, 150])

            options = {"grid_spacing": grid_spacing, "iterations": iterations, "numfi": numfi,
                       "simplex_spread": simplex_spread, "size": size, "verbose": False, "enhance_step": False,
                       "input_scale": 6}

            res = nm_gridsearch(cost_func, p0, options)

            nfev, fx, x, opt_p0 = res["nfev"], res["fun"], res["x"], res["best_start_points"][-1]
            file.write(f"{sam_idx} __ {[*np.round(x, 2)]} __ {np.round(fx, 3)} __ {[*p0]} __ "
                       f"{nfev} __ {opt_p0}\n")

            print("SamIdx: ", sam_idx, "Best minimum: ", np.round(x, 2), fx)
            sols.append(np.round(x, 2))
            fevals_all.append(nfev)

        avg_sol = f"Avg. solution: {np.mean(array(sols), axis=0)}"
        std_dev = f"Avg. std: {np.std(array(sols), axis=0)}"
        func_evals_s = f"All nfev: {fevals_all}"
        print("avg_sol: ", avg_sol)
        print("std_dev: ", std_dev)
        print("func_evals_s: ", func_evals_s)
        footer = "Avg. sol. __ std"
        file.write(footer + "\n")
        file.write(f"{avg_sol} __ {std_dev}\n")
        file.write("nfev at each grid pnt: " + func_evals_s + "\n")
        file.write("\n")
