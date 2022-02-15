import time
import numpy as np
from numba import jit
from consts import *
from results import d_best
from optimization.nelderMeadSource import _minimize_neldermead
from functions import format_data, calc_loss
from model.multir_numba import multir_numba

a, b = 0.19737935744311108, 0.300922921527581

f = array([13235.26131362, 16379.02884655, 20465.92663936,
           25181.57793875, 26753.46170521, 29897.22923814])

g = array([
    24705.82111877, 30574.18718023, 38203.06306014,
    47005.61215233, 49939.79518306, 55808.16124453])

sine_sign = lambda x: 1 if (x % (2 * pi) < pi) else -1


def sine(x):
    #x -= (x > pi) * (2 * pi)
    #x = 1.337

    B = 4 / pi
    C = -4 / (pi * pi)

    y = B * x + C * x * abs(x)

    P = 0.225
    res = P * (y * abs(y) - y) + y

    return res

def cose(x):
    x += pi/2
    x -= (x > pi)*(2*pi)

    return sine(x)


def correct_mod(s):
    return s % (2 * pi) - pi


def c_mod(s):
    res = s - 2*pi*(int(s / (2*pi)) - (s < 0)) - pi
    return res


def test_mod(s):
    if (-pi < s) and (s < pi):
        return s

    if s > 0:
        while s > 2*pi:
            s -= 2*pi
    else:
        while s < -2*pi:
            s += 2*pi

    if s < -pi:
        return s + 2*pi
    elif s > pi:
        return s - 2*pi
    else:
        return s

def explicit_reflectance(p):
    R = np.zeros(6)
    for i in range(6):
        if i != 2:
            pass
        f0 = f[i] * p[0]
        f1 = g[i] * p[1]
        f2 = f[i] * p[2]

        s0, s1, s2, s3 = f2 + f1 + f0, f2 - f1 - f0, f2 + f1 - f0, - f2 + f1 - f0

        s0 = c_mod(s0)
        s1 = c_mod(s1)
        s2 = c_mod(s2)
        s3 = c_mod(s3)

        ss0, ss1, ss2, ss3 = sine(s0), sine(s1), sine(s2), sine(s3)
        cs0, cs1, cs2, cs3 = cose(s0), cose(s1), cose(s2), cose(s3)

        print("ss0, ss1, ss2, ss3 =", ss0, ss1, ss2, ss3)
        print("cs0, cs1, cs2, cs3 =", cs0, cs1, cs2, cs3)
        exit()

        m_12_r = (1 - a * a) * b * (cs2 - cs1)
        m_22_r = (1 - a * a) * (cs0 - b * b * cs3)

        m_12_i = - 2 * a * (ss0 + b * b * ss3) + (a * a + 1) * b * (ss1 - ss2)
        m_22_i = (a * a + 1) * (ss0 + b * b * ss3) + 2 * a * b * (ss2 - ss1)

        #print(m_12_r, m_22_r, m_12_i, m_22_i)

        R[i] = (m_12_r * m_12_r + m_12_i * m_12_i) / (m_22_r * m_22_r + m_22_i * m_22_i)

    return R


if __name__ == '__main__':
    from functions import avg_runtime

    mask = custom_mask_420
    sample_idx = 10

    lam, R0 = format_data(mask=mask, sample_file_idx=sample_idx)

    p0 = np.array([35, 600, 35]) * um_to_m
    p0 = np.array([500, 500, 500]) * um_to_m
    R_numba = multir_numba(lam, p0)
    R_explicit = explicit_reflectance(p0)
    #exit()

    #avg_runtime(multir_numba, lam, d_best)
    #avg_runtime(explicit_reflectance, d_best)

    print(R_numba)
    print(R_explicit)
    #print(R_numba-R_explicit)
    print(np.all(np.isclose(R_numba, R_explicit)))

    print(calc_loss(p0, mask=mask, sample_file_idx=sample_idx))

    d0 = array([50, 600, 50]) * um_to_m
    lb = d0 - array([50, 100, 50]) * um_to_m
    hb = d0 + array([50, 100, 50]) * um_to_m

    # avg_runtime(minimize, error, d0, bounds=list(zip(lb, hb)), method='Nelder-Mead')
    error = lambda p: sum((explicit_reflectance(p) - R0)**2)
    fval, x, iterations, fcalls = _minimize_neldermead(error, d0, bounds=(lb, hb), adaptive=False)

    print(fval, x*um, iterations, fcalls)