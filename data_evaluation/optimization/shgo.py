from scipy.optimize import shgo
import numpy as np
from consts import THz, array
from nelder_mead_nD import Cost
from numpy.random import uniform

freqs = array([0.040, 0.080, 0.150, 0.550, 0.640, 0.760]) * THz

# p_sol = array([47, 640, 74.], dtype=float)
p_sol = [uniform(50, 350), uniform(400, 700), uniform(50, 350)]

new_cost = Cost(freqs, array(p_sol, dtype=float))
func = new_cost.cost

bounds = [(50, 350), (400, 700), (50, 350)]
res = shgo(func, bounds, n=350, iters=2)
#print(res.xl)
print(res.x)
print(p_sol)
