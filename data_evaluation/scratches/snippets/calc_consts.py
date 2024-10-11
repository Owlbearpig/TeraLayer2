from base_converters import dec_to_twoscompl
from consts import um_to_m
from numpy import array

a, b = 0.19737935744311108, 0.300922921527581

f = array([13235.26131362, 16379.02884655, 20465.92663936,
           25181.57793875, 26753.46170521, 29897.22923814])
g = array([
    24705.82111877, 30574.18718023, 38203.06306014,
    47005.61215233, 49939.79518306, 55808.16124453])

for g_ in g:
    g_ *= um_to_m
    i, p = 0, 20
    print(dec_to_twoscompl(g_, pd=i, p=p, format=True), f'{g_}')

for f_ in f:
    f_ *= um_to_m
    i, p = 0, 20
    print(dec_to_twoscompl(f_, pd=i, p=p, format=True), f'{f_}')

c0 = (1 - a * a) * b
c1 = (1 - a * a)
c2 = (a * a + 1) * b
c3 = - 2 * a
c4 = (a * a + 1)
c5 = 2 * a * b
c6 = - 2 * a * b * b
c7 = - (1 - a * a) * b * b
c8 = (a * a + 1) * b * b

"""
m_12_r = c0 * (cs2 - cs1)
m_22_r = c1 * cs0 + c7 * cs3

m_12_i = c3 * ss0 + c6 * ss3 + c2 * (ss1 - ss2)
m_22_i = c4 * ss0 + c8 * ss3 + c5 * (ss2 - ss1)
"""

cnst_lst = [c0, c1, c2, c3, c4, c5, c6, c7, c8]

for cnst in cnst_lst:
    i, p = 3, 17
    print(dec_to_twoscompl(cnst, pd=i, p=p), f'{cnst}')

# calc current data:
# [421. 521. 651. 801. 851. 951.] GHz, sample_idx 10, custom_mask_420
R0 = [0.01619003, 0.3079267,  0.11397636, 0.13299026, 0.05960753, 0.08666484]

for r0 in R0:
    i, p = 3, 17
    print(dec_to_twoscompl(r0, pd=i, p=p, format=True), f'{r0}')