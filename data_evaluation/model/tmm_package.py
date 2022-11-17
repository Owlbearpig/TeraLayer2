from __future__ import division, print_function, absolute_import

from tmm import coh_tmm
from scipy.constants import c as c0
from numpy import pi, linspace, inf, array


# list of layer thicknesses in um
d_list = [inf, 193.0, 544.0, 168.0, inf]

# list of refractive indices
n_list = [1, 1.50, 2.80, 1.50, 1]

freqs = array([420, 520, 650, 800, 850, 950]) * 1e9
lams = (c0 / freqs) * 10 ** 6

for lam in lams:
    r = coh_tmm("s", n_list, d_list, 0, lam)["r"]
    print(r)

