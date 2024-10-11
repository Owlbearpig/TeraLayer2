import matplotlib.pyplot as plt
import numpy as np
from numfi import numfi as numfi_
from consts import *
from matplotlib.widgets import Slider
from model.cost_function import Cost
from RTL_sim.twos_compl_OF_v2 import CostFuncFixedPoint


def main():
    noise_scale = 0.00

    # p_sol = array([282.0, 536.0, 98.0])
    # p_sol = array([200., 100.,  200.])
    p_sol = array([297.0, 619.0, 50.0])
    p_sol = array([241., 661., 237.])
    x = np.linspace(1, 1050, 500)

    #cost_func = Cost(p_solution=p_sol, noise_std_scale=noise_scale).cost
    pd, p = 4, 13
    cost_func_opts = {"pd": pd, "p": p, "use_real_data": False, "noise": noise_scale, "en_plt": True}
    cost_func = CostFuncFixedPoint(cost_func_opts).cost

    x = numfi_(x / (2*pi*2**6), s=1, w=pd + p, f=p, fixed=True, rounding='floor')
    p_sol = numfi_(p_sol / (2*pi*2**6), s=1, w=pd + p, f=p, fixed=True, rounding='floor')

    y1, y2, y3 = [], [], []
    for d in x:
        p = p_sol.copy()
        p[0] = d
        y1.append(cost_func(p))
    for d in x:
        p = p_sol.copy()
        p[1] = d
        y2.append(cost_func(p))
    for d in x:
        p = p_sol.copy()
        p[2] = d
        # p[1] = 450
        y3.append(cost_func(p))

    x = np.array(x) * (2*pi*2**6)
    plt.plot(x, y1, label="y1")
    plt.plot(x, y2, label="y2")
    plt.plot(x, y3, label="y3")
    plt.legend()


if __name__ == '__main__':
    main()
    plt.show()