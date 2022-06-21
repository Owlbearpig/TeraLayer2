import numpy as np
from numba import jit
from consts import *

from numpy import cos, sin, exp, array, arcsin, pi, conj, sum
from functions import format_data, format_data_avg, lam_axis, get_phase_measured
from model.multir_numba import multir_numba


class ExplicitEval:
    def __init__(self, data_mask, sample_file_idx=0, enable_avg=False):
        if enable_avg:
            self.lam, self.R0 = format_data_avg(data_mask)
        else:
            self.lam, self.R0 = format_data(data_mask, sample_file_idx)
            print(f'Idx of selected sample: {sample_file_idx}')

        f, r, b, s = get_phase_measured(sample_file_idx=sample_file_idx, mask=data_mask)
        self.freqs, self.ref_phase, self.bg_phase, self.sam_phase = f, r, b, s

        self.s_consts = self.set_semi_consts()
        self.unit_scale_factor = 1

        print(f'\nMeasured reflectance (R0):  {self.R0}')

    def set_R0(self, p0):
        self.R0 = self.explicit_reflectance(p0)
        print(f'\nSet new R0 to: {self.R0}\n')

    def set_semi_consts(self):
        """
        some values only need to be calculated once, they are indep. of p
        :return: None
        """
        self.the = array([thea, 0, 0, 0, 0])
        for i in range(0, 4):
            self.the[i + 1] = arcsin(n[i] * sin(self.the[i]) / n[i + 1])

        return *list(map(self.a, range(4))), \
               *list(map(self.b, range(4))), *list(map(self.f, range(0, 3)))

    def a(self, k):
        enumerator = n[k] * cos(self.the[k + 1]) - n[k + 1] * cos(self.the[k])
        denum = n[k + 1] * cos(self.the[k]) + n[k] * cos(self.the[k + 1])

        return enumerator / denum

    def b(self, k):
        enumerator = n[k + 1] * cos(self.the[k]) - n[k] * cos(self.the[k + 1])
        denum = n[k] * cos(self.the[k + 1]) + n[k + 1] * cos(self.the[k])

        return enumerator / denum

    def c(self, k):
        enumerator = 2 * n[k] * cos(self.the[k + 1])
        denum = n[k + 1] * cos(self.the[k]) + n[k] * cos(self.the[k + 1])

        return enumerator / denum

    def d(self, k):
        enumerator = 2 * n[k + 1] * cos(self.the[k])
        denum = n[k] * cos(self.the[k + 1]) + n[k + 1] * cos(self.the[k])

        return enumerator / denum

    def f(self, k):
        """
        calculate unsigned exponents
        :param k: di index
        :return: array with length == len(lam)
        """

        return 1j * 2 * pi * n[k + 1] / self.lam

    def explicit_reflectance(self, p, return_magn=True, return_r=False):
        return self.calculation(p * self.unit_scale_factor, self.s_consts, return_magn, return_r)

    def error(self, p):
        R = self.explicit_reflectance(p)
        return sum((R-self.R0)*(R-self.R0))

    def error_phase(self, p):
        _, _, _, _, phi5, phi6 = self.explicit_reflectance(p, return_magn=False)
        phi1_sam, phi2_sam, phi3_sam, phi4_sam = self.sam_phase[0:4]
        phi1_ref, phi2_ref, phi3_ref, phi4_ref = self.ref_phase[0:4]
        phi1, phi2, phi3, phi4 = phi1_sam-phi1_ref, phi2_sam-phi2_ref, phi3_sam-phi3_ref, phi4_sam-phi4_ref

        return sum((phi6 - (phi1 - phi2))**2 + (phi5 - (phi3 - phi4))**2)

    def combined_error(self, p):
        return np.log10(self.error(p)) + np.log10(self.error_phase(p))

    @staticmethod
    #@jit(cache=True, nopython=True)
    def calculation(p, s_consts, return_magn=True, return_r=False):
        """
        Note: g(k) = c(k) * d(k) - a(k) * b(k) in paper calculation is always 1.
        :param p: parameters
        :param s_consts: values which only need to be evaluated once. (Independent of p)
        :return: wavelength resolved reflectance R
        """
        a0, a1, a2, a3, b0, b1, b2, b3, f0_0, f0_1, f0_2 = s_consts
        f0, f1, f2 = p[0] * f0_0, p[1] * f0_1, p[2] * f0_2

        # the 8 terms of M_12
        t0_12 = b0 * exp(-f2 - f1 - f0)
        t1_12 = -a2 * b3 * b0 * exp(f2 - f1 - f0)
        t2_12 = b1 * exp(-f2 - f1 + f0)
        t3_12 = -a2 * b3 * b1 * exp(f2 - f1 + f0)
        t4_12 = -a1 * b0 * b2 * exp(-f0 - f2 + f1)
        t5_12 = b2 * exp(-f2 + f1 + f0)
        t6_12 = -a1 * b0 * b3 * exp(-f0 + f2 + f1)
        t7_12 = b3 * exp(f2 + f1 + f0)

        # the 8 terms of M_22
        t0_22 = -a3 * b0 * exp(-f1 - f0 - f2)
        t1_22 = -b1 * a3 * exp(-f1 + f0 - f2)
        t2_22 = -a2 * b0 * exp(f2 - f0 - f1)
        t3_22 = -a2 * b1 * exp(f2 - f1 + f0)
        t4_22 = a1 * a3 * b2 * b0 * exp(-f0 - f2 + f1)
        t5_22 = -a3 * b2 * exp(-f2 + f0 + f1)
        t6_22 = exp(f2 + f1 + f0)  # weird term
        t7_22 = -a1 * b0 * exp(f2 + f1 - f0)

        m_12 = t0_12 + t1_12 + t2_12 + t3_12 + t4_12 + t5_12 + t6_12 + t7_12
        m_22 = t0_22 + t1_22 + t2_22 + t3_22 + t4_22 + t5_22 + t6_22 + t7_22

        r = m_12 / m_22

        if return_r:
            return r

        if return_magn:
            return (r * conj(r)).real
        else:
            return np.angle(r)


if __name__ == '__main__':
    """
    Problem: Runs slower for large frequency arrays compared to old implementation, 
    even though for low frequency count (6 freqs.) it's faster... why? Array calculations?
    """
    from results import d_best
    from functions import avg_runtime
    from explicitEvalOptimized import explicit_reflectance

    explicit_reflectance.__name__ = 'original explicit_reflectance'

    mask = custom_mask_420
    lam, R = format_data(mask)

    new_eval = ExplicitEval(mask)

    d_custom1 = array([44, 680, 36]) * um_to_m
    d_custom2 = array([36, 680, 44]) * um_to_m
    d_custom3 = array([44, 36, 680]) * um_to_m

    print(new_eval.explicit_reflectance(d_custom1))
    print(new_eval.explicit_reflectance(d_custom2))
    print(new_eval.explicit_reflectance(d_custom3), "\n")

    multir_numba(lam, d_best)
    explicit_reflectance(d_best)

    avg_runtime(multir_numba, lam, d_best)
    avg_runtime(explicit_reflectance, d_best)
    new_eval.explicit_reflectance(d_best)
    avg_runtime(new_eval.explicit_reflectance, d_best)
