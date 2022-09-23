import matplotlib.pyplot as plt
from consts import custom_mask_420, um_to_m, THz, GHz, um
import numpy as np
from numpy import array, sum
from model.tmm import get_amplitude, get_phase, thickest_layer_approximation
from model.refractive_index import get_n
import matplotlib as mpl
from nelder_mead_nD import Point
from model.cost_function import Cost

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


def rand_sol():
    return [int(i) for i in [uniform(20, 300), uniform(500, 700), uniform(50, 300)]]


def is_success(sol, p):
    limit = 5
    return all([abs(sol[i] - p[i]) < limit for i in range(len(sol))])


if __name__ == '__main__':
    from numpy.random import uniform

    np.random.seed(420)
    rand = np.random.random

    all_freqs = np.arange(0.001, 1.400 + 0.001, 0.001) * THz

    test_values = []
    for _ in range(100):
        test_values.append(rand_sol())

    deviations, failures, fevals_all = [], 0, []
    with open("solutions_new.txt", "a") as file:
        header = "truth __ found __ log(fx) __ p0 __ success? __ fevals"
        file.write(header + "\n")
        for test_value in test_values:
            p_sol = array(test_value, dtype=float)

            print("Solution: ", p_sol)
            freqs = array([0.040, 0.080, 0.150, 0.550, 0.640, 0.760]) * THz  # pretty good
            # freqs = array([0.020, 0.060, 0.150, 0.550, 0.640, 0.760]) * THz
            # freqs = array([0.040, 0.080, 0.150, 0.550, 0.720, 0.780]) * THz  # pretty good

            new_cost = Cost(freqs, p_sol)

            cost_func = new_cost.cost

            p0 = array([150, 600, 150])

            grid_spacing = 50

            from scipy.optimize import basinhopping, shgo

            bounds = [(20, 300), (500, 700), (50, 300)]
            # res = basinhopping(new_cost.cost, p0, niter=50, T=1, stepsize=grid_spacing, disp=True,
            #                   minimizer_kwargs={"bounds": bounds})
            res = shgo(new_cost.cost, bounds=bounds, n=200, iters=5)

            success = is_success(res.x, p_sol)
            failures += not success

            nfev, fx = res["nfev"], res["fun"]
            file.write(f"{[*p_sol]} __ {[*np.round(res.x, 2)]} __ {round(fx, 3)} __ {[*p0]} __ "
                       f"{success} __ {nfev}\n")

            print("Best minimum: ", np.round([*res.x], 2), res["fun"])
            deviations.append(sum([abs(res.x[i] - p_sol[i]) for i in range(len(p_sol))]))
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
