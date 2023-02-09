from __future__ import division, print_function, absolute_import

import numpy as np

from consts import *
from tmm import coh_tmm
from scipy.constants import c as c0
from numpy import pi, linspace, inf, array

# example
"""
# list of layer thicknesses in um
d_list = [inf, 193.0, 544.0, 168.0, inf]

# list of refractive indices
n_list = [1, 1.50, 2.80, 1.50, 1]

freqs = array([420, 520, 650, 800, 850, 950]) * 1e9
lams = (c0 / freqs) * 10 ** 6

for lam in lams:
    r = coh_tmm("s", n_list, d_list, 0, lam)["r"]
    print(r)
"""


def check_ri(n_lst):
    n_lst = np.array(n_lst)
    if not np.isclose(n_lst[0], 1):
        n_lst = array([1, *n_lst])
    if not np.isclose(n_lst[-1], 1):
        n_lst = array([*n_lst, 1])

    return n_lst


def tmm_package_wrapper(freqs, d_list, n):
    # freq should be in THz ("between 0 and 10 THz"), d in um (wl in um)
    # n[freq_idx, n_idx]
    if d_list[0] != inf:
        d_list = [inf, *d_list]
    if d_list[-1] != inf:
        d_list = [*d_list, inf]

    angle_in = 8 * pi / 180

    if n.ndim == 1:
        lambda_vac = (c0 / freqs) * 10 ** -6
        n_list = n
        n_list = check_ri(n_list)
        r = coh_tmm("s", n_list, d_list, angle_in, lambda_vac)["r"]
        ret = array([freqs, r])
    else:
        lambda_vacs = (c0 / freqs) * 10 ** -6
        r_list = []
        for i, lambda_vac in enumerate(lambda_vacs):
            n_list = n[i]
            n_list = check_ri(n_list)
            r_list.append(coh_tmm("s", n_list, d_list, angle_in, lambda_vac)["r"])
        r_arr = array(r_list)
        ret = array([freqs, r_arr]).T

    ret = np.nan_to_num(ret)

    return ret
