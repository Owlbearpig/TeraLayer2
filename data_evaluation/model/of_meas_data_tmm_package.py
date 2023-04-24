import matplotlib.pyplot as plt

from consts import selected_freqs, array, THz, pi, c0, thea

import numpy as np
from tmm_package import coh_tmm_slim_unsafe
from meas_eval.cw.load_data import mean_data
from sample_coefficients import n


def real_data_cw(sam_idx=10):
    t_func_fd = mean_data(sam_idx, ret_t_func=True)
    freq_idx_lst = []
    for freq in selected_freqs:
        f_idx = np.argmin(np.abs(freq - t_func_fd[:, 0].real))
        freq_idx_lst.append(f_idx)

    return t_func_fd[freq_idx_lst, 1]


class CostTMM:
    def __init__(self, sam_idx=None):
        self.n = n

        r_exp = real_data_cw(sam_idx)
        print("!!! Using experimental data !!!")
        print(f"r_target: {r_exp}")

        self.r_exp_real = r_exp.real
        self.r_exp_imag = r_exp.imag

        plt.figure("Measurement")
        plt.plot(selected_freqs, np.abs(r_exp), label=f"meas amplitude {sam_idx}")
        plt.plot(selected_freqs, np.angle(r_exp), label=f"meas phase {sam_idx}")

    def cost(self, point, ret_mod=False):
        def calc_cost(p_):
            r_mod = np.zeros_like(selected_freqs, dtype=complex)
            for f_idx, freq in enumerate(selected_freqs):
                lam_vac = 10 ** 6 * c0 / (freq * 10 ** 12)
                d = array([np.inf, *p_, np.inf], dtype=float)
                r_mod[f_idx] = -1 * coh_tmm_slim_unsafe("s", self.n[f_idx], d, thea, lam_vac)

            real_error = (r_mod.real - self.r_exp_real) ** 2
            imag_error = (r_mod.imag - self.r_exp_imag) ** 2

            if ret_mod:
                return r_mod

            return np.sum(real_error + imag_error)

        try:
            x = point.x
            point.fx = calc_cost(x)
        except AttributeError:
            p = point.copy()
            return calc_cost(p)


if __name__ == '__main__':
    import time

    sam_idx = 45
    p_test = array([40, 663, 70], dtype=float)

    print("test_point: ", p_test)

    cost_inst = CostTMM(sam_idx)

    r_mod = cost_inst.cost(p_test, ret_mod=True)

    plt.plot(selected_freqs, np.abs(r_mod), label=f"model amplitude")
    plt.plot(selected_freqs, np.angle(r_mod), label=f"model phase")

    start = time.process_time()
    loss = cost_inst.cost(p_test)
    print(loss)

    plt.legend()
    plt.show()
