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


def is_success(sol, p):
    limit = 15
    return all([abs(sol[i] - p[i]) < limit for i in range(len(sol))])


if __name__ == '__main__':
    seed = 420  # generate solutions seed
    # grid options
    grid_spacing = 40
    size = 3  # 3 working but slow
    # nm options
    simplex_spread = 40  # 80  # 40 working but slow # TODO try larger bounds with model data
    iterations = 10
    # noise options
    noise_factor = 0.00

    pd, p = 4, 12

    cost_func_opts = {"pd": pd, "p": p, "use_real_data": False, "noise": noise_factor, "en_plt": False}

    dir_ = Path("results") / Path(f"FP_pd{pd}_p{p}_real_data")
    dir_.mkdir(exist_ok=True)
    numfi = partial(numfi_, s=1, w=pd + p, f=p, fixed=True, rounding='floor')

    cnt = 10
    test_values = gen_p_sols(cnt=cnt, seed=seed, layer_cnt=2)
    # test_values = cnt*[[46.0, 660.0, 76.0]]
    deviations, failures, fevals_all = [], 0, []
    # with open(dir_ / f"FP_results_nm_grid_real_data_v2.2.txt", "a") as file:
    with open(dir_ / f"FP_results_nm_sim_2layer_v1.0.txt", "a") as file:
        description = f"FP_nm, "
        description += f"Seed={seed}, iters={iterations}, size={size}, grid_spacing={grid_spacing}, pd={pd}, p={p}"
        description += f", simplex_spread={simplex_spread}, noise_factor={noise_factor}"
        header = description + "\ntruth __ found __ fx __ p0 __ success? __ fevals __ opt_p0"
        file.write(header + "\n")

        for test_idx, p_sol in enumerate(test_values):
            print(f"Sam idx {test_idx}/{cnt}")
            cost_func_opts["p_sol"] = p_sol
            cost_func_opts["sam_idx"] = test_idx
            cost_func = CostFuncFixedPoint(cost_func_opts).cost

            p0 = array([150, 600, 150])

            # bounds = [(20, 300), (500, 700), (50, 300)]
            # res = basinhopping(new_cost.cost, p0, 50, 1, grid_spacing, {"bounds": bounds}, disp=True)
            # res = shgo(cost_func, bounds=bounds, n=300, iters=5, minimizer_kwargs={"method": "Nelder-Mead"})
            options = {"grid_spacing": grid_spacing, "iterations": iterations, "numfi": numfi,
                       "simplex_spread": simplex_spread, "size": size, "verbose": False, "enhance_step": False,
                       "input_scale": 6}

            # res = nm_gridsearch(cost_func, p0, options)
            bounds = array([(0, 200.0), (500.0, 700.0), (0, 200.0)], dtype=float) / (2 * pi * 2 ** 6)
            res = shgo(cost_func, bounds, iters=3, options={"f_min": 0.1})

            success = is_success(res["x"], p_sol)
            failures += not success

            nfev, fx, x = res["nfev"], res["fun"], res["x"],
            if all(x < 3):
                x *= (2 * pi * 2 ** 6)
            if "best_start_points" in res.keys():
                opt_p0 = res["best_start_points"][-1]
            else:
                opt_p0 = None

            file.write(f"{[*p_sol]} __ {[*np.round(x, 2)]} __ {np.round(fx, 3)} __ {[*p0]} __ "
                       f"{success} __ {nfev} __ {opt_p0}\n")

            print("Solution: ", p_sol, "Best minimum: ", np.round(x, 2), fx)
            deviations.append(sum([abs(x[i] - p_sol[i]) for i in range(len(p_sol))]))
            fevals_all.append(nfev)

        avg_dev_s = f"Avg. deviation: {np.mean(array(deviations))}"
        fail_cnt_s = f"Fail count: {failures}"
        func_evals_s = f"All nfev: {fevals_all}"
        print(avg_dev_s)
        print(fail_cnt_s)
        print(func_evals_s)
        footer = "Avg. dev. __ fails"
        file.write(footer + "\n")
        file.write(f"{np.mean(array(deviations))} __ {failures}\n")
        file.write("nfev at each grid pnt: " + func_evals_s + "\n")
        file.write("\n")
