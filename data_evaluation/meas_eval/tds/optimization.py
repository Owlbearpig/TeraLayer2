import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize, shgo
from functools import partial
from cost import Cost
from consts import *


def optimize_thicknesses(d0, p0, freq_idx_range, bounds):
    best_fit, best_fit_val = None, 0
    for i in range(-5, 6):
        for j in range(-5, 6):
            print(i, j)
            d_lst = [d0[0] + i, d0[1], d0[2] + j]
            cost_inst = Cost(p0=p0, d_lst=d_lst, freq_idx_range=freq_idx_range)

            f_opt, n_opt = [], []
            for freq_idx in range(*freq_idx_range):
                cost = partial(cost_inst.cost, freq_idx=freq_idx)
                res = minimize(cost, x0=p0, bounds=bounds)

                n_opt.append(res.x)

            gof = cost_inst.gof(p=n_opt)
            if gof["peas_corr_coeff"][0] > best_fit_val:
                best_fit = d_lst
                best_fit_val = gof["peas_corr_coeff"][0]

        print(f"best_fit: {best_fit}")

        return best_fit


def main():
    f0_idx = int(0.850 / 0.014275517487508922)
    f1_idx = int(3.000 / 0.014275517487508922)
    freq_idx_range = f0_idx, f1_idx

    bounds = [(1.45, 1.55), (2.85, 2.95), (1.45, 1.55)]
    p0 = array([1.5, 2.9, 1.5])

    d_lst = [44.0, 650.0, 71.0]
    cost_inst = Cost(p0=p0, d_lst=d_lst, freq_idx_range=freq_idx_range)

    f_opt_amp, f_opt_phi, n_opt = array([]), array([]), []
    for freq_idx in range(*freq_idx_range):
        cost = partial(cost_inst.cost, freq_idx=freq_idx)
        #res = minimize(cost, x0=p0, bounds=bounds)
        res = shgo(cost, bounds=bounds)

        # print(cost_inst.freqs[freq_idx], res.x, res.fun)
        n_opt.append(res.x)
        amp_loss, phi_loss = cost(res.x, return_both=True)
        f_opt_amp = np.append(f_opt_amp, amp_loss)
        f_opt_phi = np.append(f_opt_phi, phi_loss)

    cost_inst.plot_model(p=n_opt)

    f_opt = f_opt_amp + f_opt_phi
    avg_min_cost = np.mean(f_opt)
    print(f"avg_min_cost: {avg_min_cost}")
    print(cost_inst.gof(p=n_opt))

    # cost_inst.plot_padded_n(p=n_opt)

    plt.figure("Fun val")
    plt.plot(cost_inst.freq_range, np.log10(f_opt), label="f_opt")
    plt.plot(cost_inst.freq_range, np.log10(f_opt_amp), label="f_opt_amp")
    plt.plot(cost_inst.freq_range, np.log10(f_opt_phi), label="f_opt_phi")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("log10(val)")
    plt.legend()


if __name__ == '__main__':
    main()
    plt.show()
