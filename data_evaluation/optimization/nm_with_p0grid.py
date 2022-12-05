import matplotlib.pyplot as plt
from consts import custom_mask_420, um_to_m, THz, GHz, um
import numpy as np
from numpy import array, sum
import matplotlib as mpl
from model.cost_function import Cost


if __name__ == '__main__':
    #p_sol = array([254.0, 644.0, 320.0])
    np.random.seed(420)
    #p_sol = array([100.0, 400.0, 200.0]) * (1 - np.random.random() / 10)
    p_sol = array([193.0, 544.0, 168.0])

    freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950]) * THz
    new_cost = Cost(freqs, p_sol, 0.00)
    cost_func = new_cost.cost

    p0 = array([150, 600, 150]) # shouldn´t change
    grid_spacing = 50

    from optimization.nelder_mead_nD import nm_gridsearch

    bounds = [(20, 300), (500, 700), (50, 300)]
    minimizer_kwargs = {"bounds": bounds}
    #res = basinhopping(new_cost.cost, p0, 50, 1, grid_spacing, minimizer_kwargs, disp=True)
    #res = shgo(cost_func, bounds=bounds, n=300, iters=5, minimizer_kwargs={"method": "Nelder-Mead"})
    options = {"grid_spacing" : grid_spacing, "simplex_scale": 0.80, "iterations": 15}
    res = nm_gridsearch(cost_func, p0, options)
    print(p_sol)
    print(res["x"], res["fun"])
    total_runtime = (750 / 3780) * res["total_iters"]
    print("total iterations:", res["total_iters"], f"runtime: {total_runtime} us")
    plt.plot(res["local_fun"])
    plt.show()
