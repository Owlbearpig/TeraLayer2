from numpy import array, sum, ascontiguousarray
from model.multir_numba import multir_numba
from model.explicit_eval import jacobian, reflectance
from model.explicitEvalOptimized import explicit_reflectance
from consts import um, um_to_m, custom_mask, default_mask, full_range_mask
from results import d_best
from functions import format_data, calc_loss, calc_full_loss, residuals, avg_runtime
from visualizing.plotting import plot_result
from scipy.optimize import least_squares, minimize
import scipy

lam, R = format_data(mask=default_mask, sample_file_idx=0)


def error(p):
    return sum((explicit_reflectance(p).real - R.real) ** 2)


def res_sum_jac(p):
    return sum(jacobian(p), axis=0).real


d_goal = array([0.0000378283, 0.0006273254, 0.0000378208])

d0 = array([50, 600, 50]) * um_to_m
lb = d0 - array([50, 50, 50]) * um_to_m
hb = d0 + array([50, 50, 50]) * um_to_m

bnds = list(zip(lb, hb))

#avg_runtime(minimize, error, d0, bounds=bnds, method='Nelder-Mead')
fprime = lambda x: scipy.optimize.approx_fprime(x, error, 0.01)

res = minimize(error, d0, method='Nelder-Mead')

print(res)
print(res.x * um)
print('6 freq. loss: ', calc_loss(res.x))
print('loss over full range: ', calc_full_loss(res.x), f'(d_best: {calc_full_loss(d_best)})')

plot_result(res.x, mask=full_range_mask)

