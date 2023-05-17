import matplotlib.pyplot as plt
from numpy import pi as pi64
from functions import do_fft
from scratches.snippets.base_converters import dec_to_twoscompl
from numba import jit
from model.cost_function import Cost
from consts import selected_freqs, array, THz, pi
from numfi import numfi as numfi_
from functools import partial
import numpy as np
import pandas as pan
from meas_eval.tds.main import load_data
from meas_eval.cw.load_data import mean_data
from verilog_gen import default_coeffs


def read_data_tds(sam_idx=10):
    # don't use this...
    ref_td, sam_td = load_data(sam_idx=sam_idx)

    sam_fd = do_fft(sam_td)
    ref_fd = do_fft(ref_td)

    freqs = ref_fd[:, 0].real

    closest_freqs = array([np.argmin(abs(freqs - freq)) for freq in selected_freqs])

    # plt.plot(ref_fd[:, 0], np.log10(np.abs(ref_fd[:, 1])))
    # plt.plot(ref_fd[closest_freqs, 0], np.log10(np.abs(ref_fd[closest_freqs, 1])))
    # plt.plot(sam_fd[:, 0], np.log10(np.abs(sam_fd[:, 1])))
    # plt.plot(sam_fd[closest_freqs, 0], np.log10(np.abs(sam_fd[closest_freqs, 1])))
    # plt.show()

    s, r = sam_fd[closest_freqs, 1], ref_fd[closest_freqs, 1]
    R = (s / r) ** 2
    phase_diff = np.angle(sam_fd[closest_freqs, 1]) - np.angle(ref_fd[closest_freqs, 1])

    r_exp_meas = np.sqrt(R) * np.exp(1j * phase_diff)

    return r_exp_meas


def real_data_cw(sam_idx=10):
    t_func_fd = mean_data(sam_idx, ret_t_func=True)
    freq_idx_lst = []
    for freq in selected_freqs:
        f_idx = np.argmin(np.abs(freq - t_func_fd[:, 0].real))
        freq_idx_lst.append(f_idx)

    return t_func_fd[freq_idx_lst, 1]


class CostFuncFixedPoint:
    def __init__(self, pd, p, p_sol=array([168., 609., 98.]), sam_idx=None, noise=0.0, en_plt=False):
        self.p_sol = array(p_sol)
        self.prec_int, self.prec = pd, p
        self.numfi = partial(numfi_, s=1, w=self.prec_int + self.prec, f=self.prec, fixed=True, rounding='floor')
        # self.numfi = lambda x: x

        self.freqs = selected_freqs * THz

        if sam_idx == -1:
            r_exp = Cost(freqs=self.freqs, p_solution=self.p_sol, noise_std_scale=noise, plt_mod=False).r_exp
        else:
            r_exp = real_data_cw(sam_idx)
            # r_exp = read_data_tds(sam_idx)
            # print("r_experimental:\n", r_real)
            print("!!! Using experimental data !!!")

        print(f"r_target: {r_exp}")

        self.r_exp_real = self.numfi(r_exp.real)
        self.r_exp_imag = self.numfi(r_exp.imag)

        if en_plt:
            plt.figure("Measurement")
            plt.plot(selected_freqs, np.abs(r_exp), label=f"meas amplitude {sam_idx}")
            plt.plot(selected_freqs, np.angle(r_exp), label=f"meas phase {sam_idx}")

        a, b, f, g = default_coeffs()

        # old working vals
        # a, b = 0.300922921527581, 0.19737935744311108

        # """
        # a = array([0.29682634, 0.29621877, 0.29503129, 0.29489288, 0.2947546, 0.29431419], dtype=float)
        # b = array([0.20678723, 0.20742479, 0.20901418, 0.20933128, 0.20964814, 0.21028107], dtype=float)
        # """
        self.a = self.numfi(a)
        self.b = self.numfi(b)

        # [420. 520. 650. 800. 850. 950.] GHz:
        """ # old working vals
        f = array([0.0132038236383, 0.016347591171219998, 0.02043448896403,
                   0.02515014026342, 0.02672202402988, 0.02986579156281]) * 2 ** 3
        g = array([0.024647137458149997, 0.03051550351962, 0.03814437939952,
                   0.04694692849172, 0.04988111152245, 0.055749477583909995]) * 2 ** 3
        """
        # """
        # f = array([0.01326179, 0.01644125, 0.02061997, 0.02539526, 0.02700035, 0.03021685], dtype=float) * 2 ** 3
        # g = array([0.02445803, 0.03028137, 0.03787899, 0.04663709, 0.04956974, 0.05542141], dtype=float) * 2 ** 3
        # """
        self.f, self.g = self.numfi(f * 2 ** 3), self.numfi(g * 2 ** 3)

        self.pi = self.numfi(pi64)
        self.pi2 = self.numfi(2 * pi64)
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

        self.wide_zero = numfi_(0.0, s=1, w=10 + self.prec, f=self.prec, fixed=True, rounding='floor')
        self.max_loss = 0

    def cost(self, point, ret_mod=False):

        def c_mod_v(s):
            s_fp = numfi_(array(s), s=1, w=4 + self.prec, f=self.prec, fixed=True,
                          rounding='floor')  # we can store the points as 3 + p

            s_fp_long = numfi_(s_fp, s=1, w=7 + self.prec, f=self.prec, fixed=True, rounding='floor')

            s_interm = (s_fp_long << 3) - (s_fp_long << 3).astype(int)  # 3 = 6 - 3

            res = self.pi2 * self.numfi(s_interm)

            res = numfi_(res, s=1, w=7 + self.prec, f=self.prec, fixed=True, rounding='floor')

            res_out = np.zeros_like(res)
            for i in range(len(res_out)):
                res0 = res[i]
                res1 = res0 + self.pi2
                res2 = res0 - self.pi2

                if not (res0 < 0) and (res0 > self.pi):
                    res_out[i] = res2
                elif (res0 < 0) and not (res1 > self.pi):
                    res_out[i] = res1
                else:
                    res_out[i] = res0

            return self.numfi(res_out)

        def c_mod(s):
            """
            should do (s % 2pi) and if res is > pi subtract 2pi
            max in = 2**3*(2*0.02986579156281*1000 + 0.055749477583909995 * 1000) / (2*pi*2**6) =
                   = 2.297
            max out = \pm pi
            """

            # s_scaled = s / (2 * pi64 * 2 ** 5)

            s_fp = numfi_(array(s), s=1, w=4 + self.prec, f=self.prec, fixed=True,
                          rounding='floor')  # we can store the points as 3 + p

            s_fp_long = numfi_(s_fp, s=1, w=7 + self.prec, f=self.prec, fixed=True, rounding='floor')

            s_interm = (s_fp_long << 3) - (s_fp_long << 3).astype(int)  # 3 = 6 - 3

            res = self.pi2 * self.numfi(s_interm)

            res = numfi_(res, s=1, w=7 + self.prec, f=self.prec, fixed=True, rounding='floor')

            res[res < 0] += self.pi2
            res[res > self.pi] -= self.pi2

            res = self.numfi(res)

            return res

        def sine(x):

            y = x * (self.B + self.C * np.abs(x))

            res = self.P * y * (np.abs(y) - self.one) + y

            return res

        def sin_lut(s):
            # -pi < s < pi

            return

        def cose(x):
            x += 0.5 * self.pi
            x -= (x > self.pi) * self.pi2

            return sine(x)

        def calc_cost(p_):
            f0 = self.f * self.numfi(p_[0])
            f1 = self.g * self.numfi(p_[1])
            f2 = self.f * self.numfi(p_[2])

            s0, s1, s2, s3 = f0 + f1 + f2, f1, f2 - f0, f1 - f0 - f2

            s0_, s1_, s2_, s3_ = c_mod(s0), c_mod(s1), c_mod(s2), c_mod(s3)
            s0v_, s1v_, s2v_, s3v_ = c_mod_v(s0), c_mod_v(s1), c_mod_v(s2), c_mod_v(s3)
            ss0, ss1, ss2, ss3 = sine(s0_), sine(s1_), sine(s2_), sine(s3_)
            cs0, cs1, cs2, cs3 = cose(s0_), cose(s1_), cose(s2_), cose(s3_)

            # ss0, ss1, ss2, ss3 = [np.sin(s) for s in [s0, s1, s2, s3]]
            # cs0, cs1, cs2, cs3 = [np.cos(s) for s in [s0, s1, s2, s3]]

            # sync
            d0 = ss1 * cs2
            d1 = ss1 * ss2
            # d2 = self.c4 * cs3
            # d3 = self.c4 * ss3
            d4 = self.c1 * ss0
            d5 = self.c3 * ss3

            # async
            b0 = self.c4 * cs3 - cs0
            b1 = self.c4 * ss3 + ss0
            # b0 = d2 - cs0
            # b1 = d3 + ss0
            b2_ = self.c2 * d0
            b3_ = self.c0 * d1
            b4_ = self.c6 * b1
            b5_ = self.c7 * d0

            m01_r = b3_  # 2
            m01_i = d4 + b2_ + d5  # 4

            m11_r = self.c5 * b0  # 2
            m11_i = b4_ + b5_  # 4

            # sync
            r_mod_enum_r = m01_r * m11_r + m01_i * m11_i
            r_mod_enum_i = m01_i * m11_r - m01_r * m11_i
            r_mod_denum = m11_r * m11_r + m11_i * m11_i

            # print("r_twos_compl:\n", (array(r_mod_enum_r) + 1j * array(r_mod_enum_i)) / array(r_mod_denum))
            diff_a0 = (r_mod_enum_r - self.r_exp_real * r_mod_denum)
            diff_a1 = (r_mod_enum_i - self.r_exp_imag * r_mod_denum)

            if ret_mod:
                return array(r_mod_enum_r / r_mod_denum) + 1j * array(r_mod_enum_i / r_mod_denum)
            # exit() # TODO !! check if algorithm compare gives same result without /2
            b0_error = diff_a0 * diff_a0
            b1_error = diff_a1 * diff_a1

            """
            zero = self.zero.copy()
            for m in range(len(self.freqs)):
                zero += amp_error[m]
                zero += phi_error[m]
            """
            error = b0_error + b1_error
            loss = np.sum(error)

            """
            if loss > self.max_loss:
                self.max_loss = loss
                print("New max loss", self.max_loss)
            """
            return loss

        try:
            x = point.x
            point.fx = calc_cost(x)
        except AttributeError:
            p_ = point.copy()
            return calc_cost(p_)

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
    pd, p = 4, 11
    from model.cost_function import Cost

    # p_sol = array([241., 661., 237.])
    # p_sol = array([43.0, 641.0, 74.0])
    # p_sol = array([146, 660, 73])
    # p_sol = array([46, 660, 73])
    # p_sol = array([42, 641, 74])
    p_sol = array([50, 450, 100])

    p_test = p_sol / (2 * pi * 2 ** 6)
    print("test_point: ", p_test)
    sam_idx_ = 42

    cost_func = CostFuncFixedPoint(pd=pd, p=p, sam_idx=sam_idx_).cost
    start = time.process_time()
    loss = cost_func(p_test)
    print(loss)

    plt.show()
    """
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
    """
