from model.initial_tests.explicitEvalOptimizedClean import ExplicitEval
from consts import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

mask = custom_mask_420
sample_idx = 0
enable_avg = False
model_calc = True

new_eval = ExplicitEval(mask, sample_file_idx=sample_idx, enable_avg=enable_avg)
new_eval.unit_scale_factor = um_to_m

p0 = array([45, 628, 45])
new_eval.set_R0(p0)


def f(lam, p0, p1, p2):
    p = array([p0, p1, p2])
    return new_eval.explicit_reflectance(p)


x_data, y_data = new_eval.lam, new_eval.R0

popt, pcov = curve_fit(f, x_data, y_data, p0=array([40, 625, 40]))

print(popt)
print(pcov)
plt.imshow(np.log10(np.abs(pcov)))
plt.colorbar()
plt.show()
