import tmm
from numpy import array
import numpy as np
from scipy.constants import c
from consts import um_to_m, GHz, selected_freqs, n0, n1, n2
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
from scipy.constants import c as c0
from tmm_package import coh_tmm_slim
from scipy.fftpack import fft, rfft, rfftfreq
from functools import partial
from helpers import multi_root

from scipy.optimize import minimize

minimize = partial(minimize, method="Nelder-Mead")

"""
f_idx   period   wavelength  1/period
0 0.0019 1/um 	2997.925	526
1 0.0046 	1199.17		217
2 0.0121 	461.219		83
3 0.0148 	374.741		68
4 0.0169 	329.442		59
6 0.0195 	285.517		51

- Taking abs value of expr1 means that minima positions do not depend on phase error.
- Amplitude error causes minima around correct value 

For 1-500 um, 1-500 um range in d1 and d2, 
we would optimize from around (500/40)^2 = 156 starting points.

IDEA:
1. For each frequency: 
0 = c0 + c1x + c2y + c3xy, x, y = e^(-i 2phi1), e^(-i 2phi2), 
where ci contain r_exp.
=> 1 = np.abs((c0 + c1 * x) / (c2 + c3 * x))

2. 0 = sum_(freqs) (1 - np.abs((c0 + c1 * x) / (c2 + c3 * x)))**2 (expr1)
=> find local minima (d1_min={D0, D1, ..., Dn}) along x(d1).
(Separation distance (starting points for optimization)
can be approximated by taking FFT of eq1.?) 

3. optimize original expression expr2: 0=(r_exp - r_mod(d1, d2))^2 with d1 fixed from d1_min.
Should be easier since d1 is already known? 

2x 1D optimization instead of direct single 2D optimization. Is it even faster?
"""

freqs = selected_freqs.copy()
lam = c0 * 1e-6 / freqs
print(f"Frequencies: {freqs} THz,\nwavelengths {np.round(lam, 3)} um")
print(f"Refractive indices: n1={n1},\nn2={n2}")
d_truth = [np.inf, 50, 285, np.inf]

r0, r1, r2 = (1 - n1) / (1 + n1), (n1 - n2) / (n1 + n2), (n2 - 1) / (n2 + 1)
# phi = (2 * np.random.random() - 1) * np.pi

d1, d2 = np.arange(1, 500, 1), np.arange(1, 500, 1)

r_exp = np.zeros(len(freqs), dtype=complex)
for freq_idx_ in range(freqs.size):
    r_exp[freq_idx_] = coh_tmm_slim("s", [1, n1[freq_idx_], n2[freq_idx_], 1], d_truth, 0, lam[freq_idx_])

r_exp_abs, r_exp_ang = np.abs(r_exp), np.angle(r_exp)
eu = np.exp(1j * r_exp_ang)

c0 = r2 - r_exp_abs * r0 * r2 * eu
c1 = r0 * r1 * r2 - r_exp_abs * r1 * r2 * eu
c2 = r1 - r_exp_abs * r0 * r1 * eu
c3 = r0 - r_exp_abs * eu


def expr1(d1_, freq_idx_=0):
    phi1 = 2 * np.pi * d1_ * n1 / lam

    x_ = np.exp(1j * 2 * phi1[freq_idx_])
    s = 1-np.sum(np.abs((c0[freq_idx_] + c1[freq_idx_] * x_) / (c2[freq_idx_] + c3[freq_idx_] * x_)))

    return abs(s)


def expr2(d1_, d2_):
    phi1, phi2 = 2 * np.pi * d1_ * n1 / lam, 2 * np.pi * d2_ * n2 / lam
    x_, y_ = np.exp(1j * 2 * phi1), np.exp(1j * 2 * phi2)

    diff = c0 + c1 * x_ + c2 * y_ + c3 * x_ * y_

    return np.sum(diff.real**2 + diff.imag**2)


f_idx = 5

Y = rfft([expr1(d1_, freq_idx_=f_idx) for d1_ in np.arange(1, 500000, 1)])
f_axis = rfftfreq(len(Y))

plt.figure()
plt.title("FFT of expr1 wrt d1")
plt.plot(f_axis[1:], np.abs(Y)[1:])


plt.figure()
plt.title("Expression 1")
plt.plot(d1, [expr1(d1_, freq_idx_=f_idx) for d1_ in d1])
plt.ylabel("Difference")
plt.xlabel("$d_1$")
plt.legend()

opt_minima_expr1 = []
step = 30
for d1_x0 in np.arange(1, 500, step):
    #expr1_ = partial(expr1, freq_idx_=1)
    opt_res_expr1 = minimize(expr1, d1_x0)
    opt_minima_expr1.append(opt_res_expr1["x"][0])
    #print(opt_res_expr1)

plt.figure()
for d1_minima in opt_minima_expr1:
    y = [expr2(d1_minima, d2_) for d2_ in np.arange(1, 1000, 1)]
    plt.plot(np.arange(1, 1000, 1), y, label=f"{d1_minima}")

opt_minima_expr2 = []
for d2_x0 in np.arange(1, 500, step):
    #expr2_ = partial(expr2, d1_=)
    #opt_res_expr2 = minimize(expr2, d2_x0)
    #opt_minima_expr2.append((d2_x0, opt_res_expr2["x"][0]))
    pass
plt.legend()
plt.show()
