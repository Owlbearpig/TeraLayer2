from numpy import array, sum, ascontiguousarray
from model.multir_numba import multir_numba
from model.explicit_eval import jacobian, reflectance
from consts import um, um_to_m, custom_mask, default_mask
from functions import format_data, calc_loss, calc_full_loss, residuals
from visualizing.plotting import plot_result
from scipy.optimize import least_squares, minimize

lam, R = format_data(mask=default_mask, sample_file_idx=0)


def res_sum(p):
    return sum(residuals(p, reflectance, lam, R)).real


def res_sum_jac(p):
    return sum(jacobian(p), axis=0).real


d_goal = array([0.0000378283, 0.0006273254, 0.0000378208])

d0=array([0.000045, 0.00060, 0.000045])
lb = array([0.000001, 0.00001, 0.000001])
hb = array([0.001, 0.001, 0.001])

bnds = list(zip(lb, hb))

res = minimize(res_sum, d0, bounds=bnds, jac=res_sum_jac, method='TNC')

print(res)
print(res.x * um)
print('6 freq. loss: ', calc_loss(res.x))
print('loss over full range: ', calc_full_loss(res.x))

plot_result(res.x, mask=default_mask)

