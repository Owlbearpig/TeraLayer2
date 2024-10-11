import numpy as np
from consts import n_, selected_freqs, c_thz, thea
from tmm import coh_tmm
from RTL_sim.twos_compl_OF_v2 import real_data_cw, CostFuncFixedPoint

from sympy import *

init_printing(use_unicode=True)

"""
#### model comparison FP vs DOUBLE
sam_idx = 20
r_exp = real_data_cw(sam_idx)

lam = c_thz / selected_freqs
d_test = np.array([42, 626, 63], dtype=float)

r_tmm = np.zeros_like(selected_freqs, dtype=complex)
for i, lam_vac in enumerate(lam):
    d_list = [np.inf, *d_test, np.inf]
    n_list = [1, *n_[i], 1]
    r_tmm[i] = -np.conj(coh_tmm("s", n_list, d_list, thea, lam_vac)["r"])

cost_func_opts = {"pd": 4, "p": 22, "use_real_data": True, "en_plt": False, "sam_idx": sam_idx}
cost_func = CostFuncFixedPoint(cost_func_opts).cost

loss_tmm = np.sum((r_tmm.real - r_exp.real) ** 2 + (r_tmm.imag - r_exp.imag) ** 2)

r_fp = cost_func(d_test / (2 * pi * 2 ** 6), ret_mod=True)
loss_fp = cost_func(d_test / (2 * pi * 2 ** 6), ret_mod=False)

print("DOUBLE IMPLEMENTATION")
print(r_tmm)
print(loss_tmm)
print("FIXED POINT IMPLEMENTATION")
print(r_fp)
print(loss_fp)

# m = Matrix([[1, -1], [3, 4], [0, 2]])
# ### model comparison FP vs DOUBLE
"""

phi1, phi2, phi3 = symbols("phi_1 phi_2 phi_3")
r0, r1, r2, r3 = symbols("r_0 r_1 r_2 r_3")

m0 = Matrix([[1, r0], [r0, 1]])
m1 = Matrix([[1, r1], [r1, 1]])
m2 = Matrix([[1, r2], [r2, 1]])
m3 = Matrix([[1, r3], [r3, 1]])

p1 = Matrix([[exp(-1j*phi1), 0], [0, exp(1j*phi1)]])
p2 = Matrix([[exp(-1j*phi2), 0], [0, exp(1j*phi2)]])
p3 = Matrix([[exp(-1j*phi3), 0], [0, exp(1j*phi3)]])

m = m0*p1*m1*p2*m2*p3*m3
print(m)
