import time
import numpy as np
from numba import jit
from consts import *
from results import d_best
from functions import format_data, calc_loss
from model.multir_numba import multir_numba

a, b = 0.19737935744311108, 0.300922921527581

f = array([13235.26131362, 16379.02884655, 20465.92663936,
           25181.57793875, 26753.46170521, 29897.22923814])

g = array([
    24705.82111877, 30574.18718023, 38203.06306014,
    47005.61215233, 49939.79518306, 55808.16124453,
])

sine_sign = lambda x: 1 if (x % (2 * pi) < pi) else -1

def explicit_reflectance(p):
    R = np.zeros(6)
    for i in range(6):
        f0 = f[i] * p[0]
        f1 = g[i] * p[1]
        f2 = f[i] * p[2]

        s0, s1, s2, s3 = f2 + f1 + f0, f2 - f1 - f0, f2 + f1 - f0, -f2 + f1 - f0

        cs0, cs1, cs2, cs3 = cos(s0), cos(s1), cos(s2), cos(s3)
        ss0, ss1, ss2, ss3 = sin(s0), sin(s1), sin(s2), sin(s3)

        m_12_r = (1 - a * a) * b * (cs2 - cs1)
        m_22_r = (1 - a * a) * (cs0 - b * b * cs3)

        m_12_i = - 2 * a * (ss0 + b * b * ss3) + (a * a + 1) * b * (ss1 - ss2)
        m_22_i = (a * a + 1) * (ss0 + b * b * ss3) + 2 * a * b * (ss2 - ss1)

        R[i] = (m_12_r * m_12_r + m_12_i * m_12_i) / (m_22_r * m_22_r + m_22_i * m_22_i)

    return R


if __name__ == '__main__':
    from functions import avg_runtime

    mask = custom_mask_420
    sample_idx = 10

    lam, R0 = format_data(mask=mask, sample_file_idx=sample_idx)

    p0 = np.array([500, 600, 25]) * um_to_m
    R_numba = multir_numba(lam, p0)
    R_explicit = explicit_reflectance(p0)

    #avg_runtime(multir_numba, lam, d_best)
    #avg_runtime(explicit_reflectance, d_best)

    print(R_numba)
    print(R_explicit)
    print(np.all(np.isclose(R_numba, R_explicit)))


    print(calc_loss(p0, mask=mask, sample_file_idx=sample_idx))
