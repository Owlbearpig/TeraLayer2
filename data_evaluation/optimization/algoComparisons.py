import matplotlib.pyplot as plt
from consts import custom_mask_420, um_to_m, THz, GHz, um
import numpy as np
from numpy import array, sum
import matplotlib as mpl
from model.cost_function import Cost
from functions import gen_p_sols

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

    test_values = gen_p_sols(cnt=100)

    deviations, failures, fevals_all = [], 0, []
    with open("results_nm_grid.txt", "a") as file:
        description = "p0GridSearch without noise, 0.80 init simplex scale, "
        description += "421 truth seed, 16 iters, size=2, spacing=55, no_div_loss + approximation"
        header = description + "\ntruth __ found __ fx __ p0 __ success? __ fevals __ opt_p0"
        file.write(header + "\n")

        for test_value in test_values:
            p_sol = array(test_value, dtype=float)

            # freqs = array([0.040, 0.080, 0.150, 0.550, 0.640, 0.760]) * THz  # pretty good
            # freqs = array([0.020, 0.060, 0.150, 0.550, 0.640, 0.760]) * THz
            # freqs = array([0.040, 0.080, 0.150, 0.550, 0.720, 0.780]) * THz  # pretty good
            freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950]) * THz  # GHz; freqs. set on fpga
            new_cost = Cost(freqs, p_sol, 0.00)
            cost_func = new_cost.cost

            p0 = array([150, 600, 150])
            grid_spacing = 55

            from scipy.optimize import basinhopping, shgo
            from optimization.nelder_mead_nD import nm_gridsearch

            bounds = [(20, 300), (500, 700), (50, 300)]
            minimizer_kwargs = {"bounds": bounds}
            # res = basinhopping(new_cost.cost, p0, 50, 1, grid_spacing, minimizer_kwargs, disp=True)
            # res = shgo(cost_func, bounds=bounds, n=300, iters=5, minimizer_kwargs={"method": "Nelder-Mead"})
            options = {"grid_spacing": grid_spacing, "simplex_scale": 0.80, "iterations": 17, "size": 2,
                       "verbose": False, "enhance_step": False}
            res = nm_gridsearch(cost_func, p0, options)

            success = is_success(res["x"], p_sol)
            failures += not success

            nfev, fx, x, opt_p0 = res["nfev"], res["fun"], res["x"], res["lstart"][-1]
            file.write(f"{[*p_sol]} __ {[*np.round(x, 2)]} __ {round(fx, 3)} __ {[*p0]} __ "
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
