import time
import numpy as np
from numba import jit
from consts import n, thea, default_mask, wide_mask, full_range_mask
from results import d_best
from numpy import cos, sin, exp, array, arcsin, pi, conj, sum, outer
from functions import format_data
from model.multir_numba import multir_numba


the = array([thea, 0, 0, 0, 0])
for i in range(0, 4):
    the[i + 1] = arcsin(n[i] * sin(the[i]) / n[i + 1])

lam, R0 = format_data(mask=default_mask)


def a(k):
    enumerator = n[k] * cos(the[k + 1]) - n[k + 1] * cos(the[k])
    denum = n[k + 1] * cos(the[k]) + n[k] * cos(the[k + 1])

    return enumerator / denum


def b(k):
    enumerator = n[k + 1] * cos(the[k]) - n[k] * cos(the[k + 1])
    denum = n[k] * cos(the[k + 1]) + n[k + 1] * cos(the[k])

    return enumerator / denum


def c(k):
    enumerator = 2 * n[k] * cos(the[k + 1])
    denum = n[k + 1] * cos(the[k]) + n[k] * cos(the[k + 1])

    return enumerator / denum


def d(k):
    enumerator = 2 * n[k + 1] * cos(the[k])
    denum = n[k] * cos(the[k + 1]) + n[k + 1] * cos(the[k])

    return enumerator / denum


def g(k):
    return c(k) * d(k) - a(k) * b(k)


def f(k):
    """
    calculate unsigned exponents
    :param k: di index
    :return: array with length == len(lam)
    """
    return 1j * 2 * pi * n[k + 1] / lam


# ai, bi const wrt to wls, thicknesses -> gi const.
g1, g2, g3 = g(1), g(2), g(3)
a0, a1, a2, a3, b0, b1, b2, b3 = a(0), a(1), a(2), a(3), b(0), b(1), b(2), b(3)

# exponents, wl resolved. Indices indicate di.
f0_0, f0_1, f0_2 = f(0), f(1), f(2)


@jit(cache=True, nopython=True)
def explicit_reflectance(p):
    f0, f1, f2 = p[0]*f0_0, p[1]*f0_1, p[2]*f0_2

    # the 8 terms of M_12
    t0_12 = g3 * g2 * g1 * b0 * exp(-f2-f1-f0)
    t1_12 = -a2 * b3 * g1 * b0 * exp(f2-f1-f0)
    t2_12 = g3 * g2 * b1 * exp(-f2-f1+f0)
    t3_12 = -a2 * b3 * b1 * exp(f2-f1+f0)
    t4_12 = -a1 * b0 * g3 * b2 * exp(-f0-f2+f1)
    t5_12 = g3 * b2 * exp(-f2+f1+f0)
    t6_12 = -a1 * b0 * b3 * exp(-f0+f2+f1)
    t7_12 = b3 * exp(f2+f1+f0)

    # the 8 terms of M_22
    t0_22 = -a3 * g2 * g1 * b0 * exp(-f1-f0-f2)
    t1_22 = -b1 * a3 * g2 * exp(-f1+f0-f2)
    t2_22 = -a2 * g1 * b0 * exp(f2-f0-f1)
    t3_22 = -a2 * b1 * exp(f2-f1+f0)
    t4_22 = a1 * a3 * b2 * b0 * exp(-f0-f2+f1)
    t5_22 = -a3 * b2 * exp(-f2+f0+f1)
    t6_22 = exp(f2+f1+f0)  # weird term
    t7_22 = -a1 * b0 * exp(f2+f1-f0)

    m_12 = t0_12 + t1_12 + t2_12 + t3_12 + t4_12 + t5_12 + t6_12 + t7_12
    m_22 = t0_22 + t1_22 + t2_22 + t3_22 + t4_22 + t5_22 + t6_22 + t7_22

    r = m_12 / m_22

    return r * conj(r)


if __name__ == '__main__':
    from functions import avg_runtime

    R_numba = multir_numba(lam, d_best)
    R_explicit = explicit_reflectance(d_best)

    avg_runtime(multir_numba, lam, d_best)
    avg_runtime(explicit_reflectance, d_best)

    print(np.all(np.isclose(R_numba, R_explicit)))
