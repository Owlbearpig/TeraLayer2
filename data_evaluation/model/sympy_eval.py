import numpy as np
from consts import n_, selected_freqs, c_thz, thea
from tmm import coh_tmm
from RTL_sim.twos_compl_OF_v2 import real_data_cw, CostFuncFixedPoint

from sympy import *
init_printing(use_unicode=True)

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



