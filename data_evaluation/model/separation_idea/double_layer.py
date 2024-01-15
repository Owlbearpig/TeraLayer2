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

from scipy.optimize import minimize as minimize_


def minimize(*args, **kwargs):
    opt_res_ = minimize_(*args, **kwargs, method="Nelder-Mead")

    return opt_res_["x"]


"""
f_idx   "freq" (1/um)  wavelength  period # REDO
0 0.0019  	2997.925	526
1 0.0046 	1199.17		217
2 0.0121 	461.219		83
3 0.0148 	374.741		68
4 0.0169 	329.442		59
6 0.0195 	285.517		51
-> linear relationship between period and wavelength.

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
d1_minima_spacing = [526, 217, 83, 68, 59, 51]  # TODO REDO

freqs = selected_freqs.copy()
lam = c0 * 1e-6 / freqs
print(f"Frequencies: {freqs} THz,\nwavelengths {np.round(lam, 3)} um")
print(f"Refractive indices: n0={n0},\nn1={n1}")
d_truth = [np.inf, 250, 350, np.inf]

r0, r1, r2 = (1 - n0) / (1 + n0), (n0 - n1) / (n0 + n1), (n1 - 1) / (n1 + 1)
# phi = (2 * np.random.random() - 1) * np.pi

d1, d2 = np.arange(1, 500, 1, dtype=float), np.arange(1, 500, 1, dtype=float)

r_exp = np.zeros(len(freqs), dtype=complex)
for freq_idx_ in range(freqs.size):
    r_exp[freq_idx_] = coh_tmm_slim("s", [1, n0[freq_idx_], n1[freq_idx_], 1], d_truth, 0, lam[freq_idx_])

r, u = np.abs(r_exp), np.exp(1j * np.angle(r_exp))

c0 = r2 - r * r0 * r2 * u
c1 = r0 * r1 * r2 - r * r1 * r2 * u
c2 = r1 - r * r0 * r1 * u
c3 = r0 - r * u


def expr1(d1_, freq_idx_=0):
    phi0 = 2 * np.pi * d1_ * n0[freq_idx_] / lam[freq_idx_]

    x_ = np.exp(1j * 2 * phi0)
    s = np.abs((c0[freq_idx_] + c1[freq_idx_] * x_) / (c2[freq_idx_] + c3[freq_idx_] * x_))

    return (1-s)**2


def expr2(d1_, d2_, f_idx_=0):
    phi0, phi1 = 2 * np.pi * d1_ * n0[f_idx_] / lam[f_idx_], 2 * np.pi * d2_ * n1[f_idx_] / lam[f_idx_]
    x_, y_ = np.exp(1j * 2 * phi0), np.exp(1j * 2 * phi1)
    x_, y_ = np.linspace(-1, 1, len(x_)), np.linspace(-1, 1, len(x_))
    x_, y_ = np.meshgrid(x_, y_)
    diff = c0[f_idx_] + c1[f_idx_] * x_ + c2[f_idx_] * y_ + c3[f_idx_] * x_ * y_
    print(c0[f_idx_], c1[f_idx_], c2[f_idx_], c3[f_idx_])
    return diff.imag


f_idx = 0

x = np.linspace(-1, 1, 1000)
y = np.linspace(-1, 1, 1000)
y = c0[f_idx] + c1[f_idx] * x + c2[f_idx] * y + c3[f_idx] * x * y
#plt.figure()
#plt.plot()

plt.figure()
X, Y = np.meshgrid(d1, d2)
vals = expr2(*[X, Y], f_idx)
plt.title(f"expr2 f{f_idx}")
plt.imshow(vals,
           extent=[d1[0], d1[-1], d2[0], d2[-1]],
           origin="lower",
           # interpolation='bilinear',
           # cmap="plasma",
           vmin=0, vmax=np.mean(vals),
           )
plt.xlabel("$d_1$")
plt.ylabel("$d_2$")


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


x0 = (d1[-1]-d1[0])/2
for freq_idx in range(len(freqs)):
    expr1_ = partial(expr1, freq_idx_=freq_idx)

    opt_res1 = minimize(expr1_, x0=x0)
    right_minima = expr1_(opt_res1[0]-10) > expr1_(opt_res1[0])
    #print(right_minima)
    #print(opt_res1)


plt.legend()
plt.show()
