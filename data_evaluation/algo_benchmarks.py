from optimization.nelderMeadSource import _minimize_neldermead
from model.initial_tests.explicitEvalOptimizedClean import ExplicitEval
from consts import *
from functions import format_data, avg_runtime


lam, R = format_data(mask=custom_mask_420)
d0 = array([50, 600, 50]) * um_to_m
lb = d0 - array([50, 50, 50]) * um_to_m
hb = d0 + array([50, 50, 50]) * um_to_m

new_eval = ExplicitEval(custom_mask_420)


fval, x, iterations, fcalls = _minimize_neldermead(new_eval.error, d0, bounds=(lb, hb), adaptive=False)
print(fval, x * um, iterations, fcalls)

avg_runtime(_minimize_neldermead, new_eval.error, d0, bounds=(lb, hb))
