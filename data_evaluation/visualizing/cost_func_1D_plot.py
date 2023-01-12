import matplotlib.pyplot as plt
import numpy as np

from consts import *
from matplotlib.widgets import Slider
from model.cost_function import Cost

#p_sol = array([282.0, 536.0, 98.0])
#p_sol = array([200., 100.,  200.])
p_sol = array([297.0, 619.0, 50.0])

cost_func = Cost(p_solution=p_sol, noise_std_scale=0.00).cost


# should be resolution of axes d1, d2, d3
rez_x, rez_y, rez_z = 200, 200, 200
# rez_x, rez_y, rez_z = 1000, 1000, 1000

lb = array([0.000001, 0.000450, 0.000001]) # realistic bounds
ub = array([0.000350, 0.000750, 0.000350])
#lb = array([0.000001, 0.000001, 0.000001])
#ub = array([0.001, 0.001, 0.001])

# initial 'full' grid matching bounds
grd_x = np.linspace(lb[0], ub[0], rez_x)
grd_y = np.linspace(lb[1], ub[1], rez_y)
grd_z = np.linspace(lb[2], ub[2], rez_z)

if __name__ == '__main__':
    x = np.linspace(1, 1050, 1000)
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
        #p[1] = 450
        y3.append(cost_func(p))


    plt.plot(x, y1, label="y1")
    plt.plot(x, y2, label="y2")
    plt.plot(x, y3, label="y3")
    plt.legend()
    plt.show()