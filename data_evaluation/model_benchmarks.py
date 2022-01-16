import matplotlib.pyplot as plt
from numpy import array
from model.multir_numba import multir_numba
from model.explicitEvalOptimizedClean import ExplicitEval
from model.explicitEvalOptimized import explicit_reflectance
from model.multir import multir
from consts import default_mask, custom_mask_420
from functions import format_data, avg_runtime

lam, R = format_data(mask=custom_mask_420)

d = array([0.0000378283, 0.0006273254, 0.0000378208])

avg_runtime(multir, lam, d)
# numba is like C implementation ?
avg_runtime(multir_numba, lam, d)
avg_runtime(explicit_reflectance, d)

new_eval = ExplicitEval(data_mask=custom_mask_420)
avg_runtime(new_eval.explicit_reflectance, d)
