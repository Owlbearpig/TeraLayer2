import matplotlib.pyplot as plt
from consts import custom_mask_420, um_to_m, THz, GHz, um
import numpy as np
from numpy import array, sum
import matplotlib as mpl
from model.cost_function import Cost
from optimization.nelder_mead_nD import nm_gridsearch

if __name__ == '__main__':
    # p_sol = array([254.0, 644.0, 320.0])
    np.random.seed(420)
    # p_sol = array([100.0, 400.0, 200.0]) * (1 - np.random.random() / 10)
    #p_sol = array([193.0, 544.0, 168.0])
    # p_sol = array([76., 530., 200.])
    #p_sol = array([168., 609., 98.])
    #p_sol = array([290.0, 658.0, 94.0])
    #p_sol = array([57.0, 601.0, 252.0])
    p_sol = array([282.0, 536.0, 98.0])

    cost_func = Cost(p_solution=p_sol, noise_std_scale=0.00).cost

    p0 = array([150, 600, 150])  # shouldn't change
    grid_spacing, size = 50, 3

    #bounds = [(20, 300), (500, 700), (50, 300)]
    #minimizer_kwargs = {"bounds": bounds}
    # res = basinhopping(new_cost.cost, p0, 50, 1, grid_spacing, minimizer_kwargs, disp=True)
    # res = shgo(cost_func, bounds=bounds, n=300, iters=5, minimizer_kwargs={"method": "Nelder-Mead"})
    options = {"grid_spacing": grid_spacing, "simplex_spread": 40, "iterations": 15, "size": size,
               "verbose": False, "enhance_step": False}
    res = nm_gridsearch(cost_func, p0, options)
    print("Truth: ", p_sol)
    print(res["x"], res["fun"])
    total_runtime = (750 / 3780) * res["total_iters"]
    print("Total iterations:", res["total_iters"], f"estimated runtime: {total_runtime} us")
    print("Total obj. func. calls: ", res["nfev"])

    plt.plot(res["local_fun"])
    plt.show()
