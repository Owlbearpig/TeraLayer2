import numpy as np
from consts import n, thea, default_mask
from results import d_best
from model.initial_tests.multir_numba import multir_numba
from numpy import cos, sin, exp, array, arcsin, pi, conj
from functions import format_data


def explicit_reflectance(lam, p=None):
    _, R0 = format_data(mask=default_mask)

    if p is None:
        p = d_best

    the = array([thea, 0, 0, 0, 0])
    for i in range(0, 4):
        the[i + 1] = arcsin(n[i] * sin(the[i]) / n[i + 1])


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
        return 1j * 2 * pi * n[k + 1] * p[k] / lam

    # ai, bi const wrt to wls, thicknesses -> gi const.
    g1, g2, g3 = g(1), g(2), g(3)
    a0, a1, a2, a3, b0, b1, b2, b3 = a(0), a(1), a(2), a(3), b(0), b(1), b(2), b(3)

    # exponents, wl resolved. Indices indicate di.
    f0, f1, f2 = f(0), f(1), f(2)


    # the 8 terms of M_12
    t0_12 = g3 * g2 * g1 * b0 * exp(-f2) * exp(-f1) * exp(-f0)
    t1_12 = -a2 * b3 * g1 * b0 * exp(f2) * exp(-f1) * exp(-f0)
    t2_12 = g3 * g2 * b1 * exp(-f2) * exp(-f1) * exp(f0)
    t3_12 = -a2 * b3 * b1 * exp(f2) * exp(-f1) * exp(f0)
    t4_12 = -a1 * b0 * g3 * b2 * exp(-f0) * exp(-f2) * exp(f1)
    t5_12 = g3 * b2 * exp(-f2) * exp(f1) * exp(f0)
    t6_12 = -a1 * b0 * b3 * exp(-f0) * exp(f2) * exp(f1)
    t7_12 = b3 * exp(f2) * exp(f1) * exp(f0)


    # the 8 terms of M_22
    t0_22 = -a3 * g2 * g1 * b0 * exp(-f1) * exp(-f0) * exp(-f2)
    t1_22 = -b1 * a3 * g2 * exp(-f1) * exp(f0) * exp(-f2)
    t2_22 = -a2 * g1 * b0 * exp(f2) * exp(-f0) * exp(-f1)
    t3_22 = -a2 * b1 * exp(f2) * exp(-f1) * exp(f0)
    t4_22 = a1 * a3 * b2 * b0 * exp(-f0) * exp(-f2) * exp(f1)
    t5_22 = -a3 * b2 * exp(-f2) * exp(f0) * exp(f1)
    t6_22 = exp(f2) * exp(f1) * exp(f0)  # weird term
    t7_22 = -a1 * b0 * exp(f2) * exp(f1) * exp(-f0)

    m_12 = t0_12 + t1_12 + t2_12 + t3_12 + t4_12 + t5_12 + t6_12 + t7_12
    m_22 = t0_22 + t1_22 + t2_22 + t3_22 + t4_22 + t5_22 + t6_22 + t7_22

    r = m_12 / m_22
    R = r * conj(r)

    """
    gradient of m12 wrt d_i
    :return: "array" of size len(p) x len(lam)
    """
    d0t0_12, d1t0_12, d2t0_12 = -t0_12*f0/p[0], -t0_12*f1/p[1], -t0_12*f2/p[2]
    d0t1_12, d1t1_12, d2t1_12 = -t1_12*f0/p[0], -t1_12*f1/p[1], t1_12*f2/p[2]
    d0t2_12, d1t2_12, d2t2_12 = t2_12*f0/p[0], -t2_12*f1/p[1], -t2_12*f2/p[2]
    d0t3_12, d1t3_12, d2t3_12 = t3_12*f0/p[0], -t3_12*f1/p[1], t3_12*f2/p[2]
    d0t4_12, d1t4_12, d2t4_12 = -t4_12*f0/p[0], t4_12*f1/p[1], -t4_12*f2/p[2]
    d0t5_12, d1t5_12, d2t5_12 = t5_12*f0/p[0], t5_12*f1/p[1], -t5_12*f2/p[2]
    d0t6_12, d1t6_12, d2t6_12 = -t6_12*f0/p[0], t6_12*f1/p[1], t6_12*f2/p[2]
    d0t7_12, d1t7_12, d2t7_12 = t7_12*f0/p[0], t7_12*f1/p[1], t7_12*f2/p[2]

    """
    gradient of m22 wrt d_i
    :return: "array" of size len(p) x len(lam)
    """
    d0t0_22, d1t0_22, d2t0_22 = -t0_22*f0/p[0], -t0_22*f1/p[1], -t0_22*f2/p[2]
    d0t1_22, d1t1_22, d2t1_22 = t1_22*f0/p[0], -t1_22*f1/p[1], -t1_22*f2/p[2]
    d0t2_22, d1t2_22, d2t2_22 = -t2_22*f0/p[0], -t2_22*f1/p[1], t2_22*f2/p[2]
    d0t3_22, d1t3_22, d2t3_22 = t3_22*f0/p[0], -t3_22*f1/p[1], t3_22*f2/p[2]
    d0t4_22, d1t4_22, d2t4_22 = -t4_22*f0/p[0], t4_22*f1/p[1], -t4_22*f2/p[2]
    d0t5_22, d1t5_22, d2t5_22 = t5_22*f0/p[0], t5_22*f1/p[1], -t5_22*f2/p[2]
    d0t6_22, d1t6_22, d2t6_22 = t6_22*f0/p[0], t6_22*f1/p[1], t6_22*f2/p[2]
    d0t7_22, d1t7_22, d2t7_22 = -t7_22*f0/p[0], t7_22*f1/p[1], t7_22*f2/p[2]

    # di m12
    d0m_12 = d0t0_12 + d0t1_12 + d0t2_12 + d0t3_12 + d0t4_12 + d0t5_12 + d0t6_12 + d0t7_12
    d1m_12 = d1t0_12 + d1t1_12 + d1t2_12 + d1t3_12 + d1t4_12 + d1t5_12 + d1t6_12 + d1t7_12
    d2m_12 = d2t0_12 + d2t1_12 + d2t2_12 + d2t3_12 + d2t4_12 + d2t5_12 + d2t6_12 + d2t7_12

    # di m22
    d0m_22 = d0t0_22 + d0t1_22 + d0t2_22 + d0t3_22 + d0t4_22 + d0t5_22 + d0t6_22 + d0t7_22
    d1m_22 = d1t0_22 + d1t1_22 + d1t2_22 + d1t3_22 + d1t4_22 + d1t5_22 + d1t6_22 + d1t7_22
    d2m_22 = d2t0_22 + d2t1_22 + d2t2_22 + d2t3_22 + d2t4_22 + d2t5_22 + d2t6_22 + d2t7_22


    def dR():
        """
        calculate gradient of reflectance. It's a fraction ... g/h ' = g'h-h'g / h^2
        :return: tuple of size (len(lam), len(lam), len(lam))
        """
        t0_dR0_enum = (d0m_12*m_22-m_12*d0m_22)*conj(m_12)
        t0_dR0_denum = m_22*m_22*conj(m_22)
        t1_dR0_enum = m_12*(conj(d0m_12)*conj(m_22)-conj(m_12)*conj(d0m_22))
        t1_dR0_denum = m_22*conj(m_22)*conj(m_22)

        dR0 = t0_dR0_enum/t0_dR0_denum + t1_dR0_enum/t1_dR0_denum

        t0_dR1_enum = (d1m_12*m_22-m_12*d1m_22)*conj(m_12)
        t0_dR1_denum = m_22*m_22*conj(m_22)
        t1_dR1_enum = m_12*(conj(d1m_12)*conj(m_22)-conj(m_12)*conj(d1m_22))
        t1_dR1_denum = m_22*conj(m_22)*conj(m_22)

        dR1 = t0_dR1_enum/t0_dR1_denum + t1_dR1_enum/t1_dR1_denum

        t0_dR2_enum = (d2m_12*m_22-m_12*d2m_22)*conj(m_12)
        t0_dR2_denum = m_22*m_22*conj(m_22)
        t1_dR2_enum = m_12*(conj(d2m_12)*conj(m_22)-conj(m_12)*conj(d2m_22))
        t1_dR2_denum = m_22*conj(m_22)*conj(m_22)

        dR2 = t0_dR2_enum/t0_dR2_denum + t1_dR2_enum/t1_dR2_denum

        return dR0, dR1, dR2

    def grad_res():
        """
        we optimize F(p) = sum res^2 = sum (R-R0)^2. dF(p0, p1, p2)/di = 2*sum((R-R0)*dRi)
        :return: (2*sum((R-R0)*dR0), 2*sum((R-R0)*dR1), 2*sum((R-R0)*dR2))
        TODO I think we want gradient wl resolved, so no sum, right? Confused as usual ...
        """
        dR0, dR1, dR2 = dR()
        return np.array([2*(R-R0)*dR0, 2*(R-R0)*dR1, 2*(R-R0)*dR2]).transpose()

    return R, grad_res()


def reflectance(lam, p):
    R, _ = explicit_reflectance(lam, p)
    return R


def jacobian(p):
    lam, R = format_data(mask=default_mask)
    _, jac = explicit_reflectance(lam, p)
    return jac



if __name__ == '__main__':
    lam, R0 = format_data(mask=default_mask)

    R_numba = multir_numba(lam, d_best)
    R_explicit = explicit_reflectance(lam, d_best)

    print(np.isclose(R_numba, R_explicit))
