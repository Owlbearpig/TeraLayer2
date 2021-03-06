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
    #print(x)
    #x -= (x > pi) * (2 * pi)
    #x = 1.337

    B = 4 / pi
    C = -4 / (pi * pi)

    y = x * (B + C * abs(x))

    #print('y', y)
    P = 0.225
    res = P * y * (abs(y) - 1) + y

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
        #print(type(p[0]))
        f0 = f[i] * p[0]
        f1 = g[i] * p[1]
        f2 = f[i] * p[2]

        s0, s1, s2, s3 = f2 + f1 + f0, f2 - f1 - f0, f2 + f1 - f0, - f2 + f1 - f0
        #print("s0, s1, s2, s3", s0, s1, s2, s3)
        s0 = c_mod(s0)
        s1 = c_mod(s1)
        s2 = c_mod(s2)
        s3 = c_mod(s3)
        #print("s0, s1, s2, s3", s0, s1, s2, s3)
        ss0, ss1, ss2, ss3 = sine(s0), sine(s1), sine(s2), sine(s3)
        cs0, cs1, cs2, cs3 = cose(s0), cose(s1), cose(s2), cose(s3)

        #print("ss0, ss1, ss2, ss3", ss0, ss1, ss2, ss3)
        #print("cs0, cs1, cs2, cs3", cs0, cs1, cs2, cs3)

        #""" #correct
        m_12_r = (1 - a * a) * b * (cs2 - cs1)
        m_22_r = (1 - a * a) * (cs0 - b * b * cs3)

        m_12_i = - 2 * a * (ss0 + b * b * ss3) + (a * a + 1) * b * (ss1 - ss2)
        m_22_i = (a * a + 1) * (ss0 + b * b * ss3) + 2 * a * b * (ss2 - ss1)

        e = (m_12_r * m_12_r + m_12_i * m_12_i)
        d = (m_22_r * m_22_r + m_22_i * m_22_i)
        #print("m_12_r, m_12_i, m_22_r, m_22_i", m_12_r, m_12_i, m_22_r, m_22_i)
        #print(f"d freq. idx: {i}: {d}")
        #print(f"1/d freq. idx: {i}: {1/d}")

        R[i] = (m_12_r * m_12_r + m_12_i * m_12_i) / (m_22_r * m_22_r + m_22_i * m_22_i)

    return R


if __name__ == '__main__':
    from functions import avg_runtime

    mask = custom_mask_420
    sample_idx = 0

    lam, R0 = format_data(mask=mask, sample_file_idx=sample_idx)

    #p0 = np.array([35, 600, 35]) * um_to_m
    #p0 = np.array([10, 750, 400]) * um_to_m
    p0 = np.array([31.162230968475342, 630.244384765625,  31.162230968475342], dtype=np.float32) * um_to_m
    #p0 = np.array([500, 500, 500]) * um_to_m
    #R_numba = multir_numba(lam, p0)
    R_explicit = explicit_reflectance(p0)

    #print(R_explicit)
    #exit()
    for i in range(0, 6):
        p = np.array([500+i*10, 500+i*20, 500+i*30]) * um_to_m
        R_ = explicit_reflectance(p)
        #print(p*10**6)
        #print(R_[0])
    #exit()
    #avg_runtime(multir_numba, lam, d_best)
    #avg_runtime(explicit_reflectance, d_best)

    #print(R_numba)
    #print(R_explicit)
    #print(R_numba-R_explicit)
    #print(np.all(np.isclose(R_numba, R_explicit)))

    #print(calc_loss(p0, mask=mask, sample_file_idx=sample_idx))

    d0 = array([50, 600, 50]) * um_to_m
    lb = d0 - array([50, 100, 50]) * um_to_m
    hb = d0 + array([50, 100, 50]) * um_to_m

    # avg_runtime(minimize, error, d0, bounds=list(zip(lb, hb)), method='Nelder-Mead')
    error = lambda p: sum((explicit_reflectance(p) - R0)**2)
    #print('error:', error(p0 + array([1, 0, 0])*um_to_m))
    #print(R0)
    fval, x, iterations, fcalls = _minimize_neldermead(error, d0, bounds=(lb, hb), adaptive=False)
    #print(fval, x*um, iterations, fcalls)

    from scratches.snippets.base_converters import dec_to_twoscompl
    dp, p = 3, 17
    print(f"R (dec):", explicit_reflectance(p0))
    print(f"R (bin):", [dec_to_twoscompl(r, dp, p) for r in explicit_reflectance(p0)])
    print(f"R0_idx{sample_idx} (dec):", R0)
    print(f"R0_idx{sample_idx} (bin):", [dec_to_twoscompl(r, dp, p) for r in R0])
    #print(f"diff^2: ", (explicit_reflectance(p0) - R0) ** 2)
    s = 0
    for i in range(6):
        R = explicit_reflectance(p0)
        s = s + (R[i] - R0[i]) ** 2
        #print(i, s)
    print('fx:', error(p0))
    print("(R-R0)**2", (explicit_reflectance(p0) - R0) ** 2)