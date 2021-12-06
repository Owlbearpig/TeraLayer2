from numpy import array, sum, ascontiguousarray
from model.multir_numba import multir_numba
from model.explicit_eval import jacobian, reflectance
from model.explicitEvalOptimized import explicit_reflectance
from consts import um, um_to_m, default_mask, full_range_mask
from results import d_best
from functions import format_data, calc_loss, calc_full_loss, residuals, avg_runtime
from visualizing.plotting import plot_result
from scipy.optimize import least_squares, minimize
from model.explicitEvalOptimizedClean import ExplicitEval
from optimization.nelderMeadSource import _minimize_neldermead
import scipy

mask = default_mask

lam, R = format_data(mask=mask, sample_file_idx=10)


def error(p):
    return sum((multir_numba(lam, p).real - R.real) ** 2)


def res_sum_jac(p):
    return sum(jacobian(p), axis=0).real


d_goal = array([0.0000378283, 0.0006273254, 0.0000378208])

d0 = array([50, 600, 50]) * um_to_m
lb = d0 - array([50, 100, 50]) * um_to_m
hb = d0 + array([50, 100, 50]) * um_to_m

#avg_runtime(minimize, error, d0, bounds=list(zip(lb, hb)), method='Nelder-Mead')
new_eval = ExplicitEval(mask)
fval, x, iterations, fcalls = _minimize_neldermead(new_eval.error, d0, bounds=(lb, hb), adaptive=False)
avg_runtime(_minimize_neldermead, error, d0, bounds=(lb, hb))
#res = minimize(error, d0, jac=res_sum_jac, method='SLSQP', bounds=list(zip(lb, hb)))
#x = res.x
#print(iterations, fcalls)
print(x * um)
print(f'{len(mask)} freq. loss: ', calc_loss(x))
print('loss over full range: ', calc_full_loss(x), f'(d_best: {calc_full_loss(d_best)})')

plot_result(x, mask=mask)
