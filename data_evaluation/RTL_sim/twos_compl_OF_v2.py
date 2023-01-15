import matplotlib.pyplot as plt
from numpy import pi as pi64
from scratches.snippets.base_converters import dec_to_twoscompl
from numba import jit
from model.cost_function import Cost
from consts import *
from numfi import numfi as numfi_
from functools import partial
import numpy as np


class CostFuncFixedPoint:
    def __init__(self, pd, p, p_sol = array([168., 609., 98.]), noise=0.0, plt_mod=False):
        self.p_sol = array(p_sol)

        self.numfi = partial(numfi_, s=1, w=pd + p, f=p, fixed=True, rounding='floor')

        self.freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950]) * THz

        r_exp = Cost(freqs=self.freqs, p_solution=self.p_sol, noise_std_scale=noise, plt_mod=plt_mod).r_exp

        self.r_exp_real = self.numfi(r_exp.real)
        self.r_exp_imag = self.numfi(r_exp.imag)

        a, b = 0.300922921527581, 0.19737935744311108

        self.a = self.numfi(a)
        self.b = self.numfi(b)

        # [420. 520. 650. 800. 850. 950.] GHz:
        f = array([0.0132038236383, 0.016347591171219998, 0.02043448896403,
                   0.02515014026342, 0.02672202402988, 0.02986579156281]) * 2**3
        g = array([0.024647137458149997, 0.03051550351962, 0.03814437939952,
                   0.04694692849172, 0.04988111152245, 0.055749477583909995]) * 2**3

        self.f, self.g = self.numfi(f), self.numfi(g)

        self.pi = self.numfi(pi64)
        self.pi2 = self.numfi(2*pi64)
        self.pi2_inv = self.numfi(1 / (2 * pi64))

        # sine consts:
        self.B = self.numfi(4 / pi64)
        self.C = self.numfi(-4 / (pi64 * pi64))
        self.P = self.numfi(0.225)

        self.zero = self.numfi(0)
        self.one, self.two, self.three, self.four = self.numfi(1), self.numfi(2), self.numfi(3), self.numfi(4)

        self.c0 = self.two * self.a * (self.b * self.b - self.one)
        self.c1 = self.two * self.b
        self.c2 = self.two * self.a * (self.one + self.b * self.b)
        self.c3 = self.two * self.a * self.a * self.b
        self.c4 = self.a * self.a
        self.c5 = self.b * self.b - self.one
        self.c6 = self.b * self.b + self.one
        self.c7 = self.four * self.a * self.b

        self.wide_zero = numfi_(0.0, s=1, w=10 + p, f=p, fixed=True, rounding='floor')
        self.max_loss = 0

    def cost(self, point):
        def c_mod(s):
            """
            should do (s % 2pi) and if res is > pi subtract 2pi
            max in = 2**3*(2*0.02986579156281*1000 + 0.055749477583909995 * 1000) / (2*pi*2**6) =
                   = 2.297
            max out = \pm pi
            """

            #s_scaled = s / (2 * pi64 * 2 ** 5)
            #
            s_fp = numfi_(array(s), s=1, w=3 + p, f=p, fixed=True, rounding='floor') # we can store the points as 3 + p

            s_fp_long = numfi_(s_fp, s=1, w=7 + p, f=p, fixed=True, rounding='floor')

            s_interm = (s_fp_long << 3) - (s_fp_long << 3).astype(int) # 3 = 6 - 3

            res = self.pi2 * self.numfi(s_interm)

            res[res < 0] += self.pi2
            res[res > self.pi] -= self.pi2

            return res

        def sine(x):

            y = x * (self.B + self.C * np.abs(x))

            res = self.P * y * (np.abs(y) - self.one) + y

            return res

        def cose(x):
            x += 0.5 * self.pi
            x -= (x > self.pi) * self.pi2

            return sine(x)

        def calc_cost(p_):
            f0 = self.f * p_[0]
            f1 = self.g * p_[1]
            f2 = self.f * p_[2]

            s0, s1, s2, s3 = f0 + f1 + f2, f1, f2 - f0, f1 - f0 - f2

            s0, s1, s2, s3 = c_mod(s0), c_mod(s1), c_mod(s2), c_mod(s3)

            ss0, ss1, ss2, ss3 = sine(s0), sine(s1), sine(s2), sine(s3)
            cs0, cs1, cs2, cs3 = cose(s0), cose(s1), cose(s2), cose(s3)

            d0 = ss1 * cs2

            m01_r = self.c0 * ss1 * ss2  # 2
            m01_i = self.c1 * ss0 + self.c2 * d0 + self.c3 * ss3  # 4

            m11_r = self.c5 * (self.c4 * cs3 - cs0)  # 2
            m11_i = self.c6 * (self.c4 * ss3 + ss0) + self.c7 * d0  # 4

            r_mod_enum_r = m01_r * m11_r + m01_i * m11_i
            r_mod_enum_i = m01_i * m11_r - m01_r * m11_i
            r_mod_denum = m11_r * m11_r + m11_i * m11_i

            amp_diff = (r_mod_enum_r - self.r_exp_real * r_mod_denum)
            phi_diff = (r_mod_enum_i - self.r_exp_imag * r_mod_denum)

            amp_error = 0.5 * amp_diff * amp_diff
            phi_error = 0.5 * phi_diff * phi_diff

            """
            zero = self.zero.copy()
            for m in range(len(self.freqs)):
                zero += amp_error[m]
                zero += phi_error[m]
            """

            loss = np.sum(amp_error + phi_error)
            """
            if loss > self.max_loss:
                self.max_loss = loss
                print("New max loss", self.max_loss)
            """
            return loss

        try:
            p = point.x
            point.fx = calc_cost(p)
        except AttributeError:
            p = point.copy()
            return calc_cost(p)

        """
        if type(point) is np.ndarray:
            p = point.copy()
            return calc_cost(p)
        else:
            p = point.x
            point.fx = calc_cost(p)
        """

if __name__ == '__main__':
    import time
    """
    // model data (r_exp) for p_sol = [168. 609.  98.], 
    // p = [239.777814149857 476.259423971176 235.382882833481] 
    // => f(p_sol, p) = 8.00341041043292 / 2 = 4.00170520521646 (python)
    // p = [999, 999, 999]
    // => f(p_sol, p) = 0.5715376463789499 (python) 
    """
    pd, p = 4, 16
    noise_factor = 0.05
    seed = 420
    from model.cost_function import Cost
    from functions import gen_p_sols

    sols = gen_p_sols(1000, seed=seed)
    for i, p_sol in enumerate(sols):
        #p_sol = array([282.0, 509.0, 50.0])
        #p_sol = array([999.0, 999.0, 999.0])

        cost_func = CostFuncFixedPoint(p_sol=p_sol, pd=pd, p=p, noise=noise_factor).cost

        p_test = p_sol / (2*pi*2**6)

        start = time.process_time()
        loss = cost_func(p_test)

        if loss > 0.001:
            print("Runtime: ", time.process_time() - start, "(s)")
            print(p_sol)
            print(loss)
        else:
            print("passed", i)
