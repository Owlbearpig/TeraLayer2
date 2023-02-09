import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize, shgo, basinhopping
from functools import partial
from cost import Cost
from consts import *


def optimize_thicknesses(d0, p0, freq_idx_range, bounds):
    best_fit_pcc, max_val_pcc = None, 0
    best_fit_loss, min_val_loss = None, np.inf
    m = freq_idx_range[1] - freq_idx_range[0]  # freq_cnt
    for x in range(-5, 6):
        for y in range(-5, 6):
            z = -5
            print(x, y, z)
            d_lst = [d0[0] + x, d0[1] + y, d0[2] + z]
            cost_inst = Cost(p0=p0, d_lst=d_lst, freq_idx_range=freq_idx_range)

            f_opt, n_opt = np.zeros(m), np.zeros((m, len(bounds)))
            for loop_idx, freq_idx in enumerate(range(*freq_idx_range)):
                cost = partial(cost_inst.cost, freq_idx=freq_idx)
                # res = minimize(cost, x0=p0, bounds=bounds)
                res = shgo(cost, bounds=bounds, iters=4)

                n_opt[loop_idx] = res.x
                f_opt[loop_idx] = res.fun

            gof = cost_inst.gof(p=n_opt)
            if gof["peas_corr_coeff"][0] > max_val_pcc:
                best_fit_pcc = d_lst
                max_val_pcc = gof["peas_corr_coeff"][0]

            avg_min_cost = np.mean(f_opt)
            if avg_min_cost < min_val_loss:
                best_fit_loss = d_lst
                min_val_loss = avg_min_cost

        print(f"best_fit_pcc: {best_fit_pcc}")
        print(f"best_fit_loss: {best_fit_loss}")

        return best_fit_pcc, best_fit_loss


def optimize_sm(d0, p0, freq_idx_range, bounds):
    cost_inst = Cost(p0=p0, d_lst=d0, freq_idx_range=freq_idx_range)

    cost = cost_inst.cost_sm

    res = shgo(cost, bounds=bounds)
    print(res)
    print(cost_inst.gof(p=res.x, sm=True))

    cost_inst.plot_model(p=res.x, sm=True)
    cost_inst.plot_n(p=res.x, sm=True)


def main():
    df = 0.014275517487508922
    f0_idx = int(0.150 / df)
    f1_idx = int(4.000 / df)

    freq_idx_range = f0_idx, f1_idx

    bounds = array([(1.45, 1.55), (2.85, 2.95), (1.45, 1.55)])
    bounds = array([(1.40, 1.60), (2.80, 3.00), (1.40, 1.60), (0.00, 0.05)])

    array([0.4, 0.06, 0.4, 2.50e-3, 0.8e-3, 40,
           1.4, 0.55, 5.0, 5.5e-3, 1.3e-2, 310,
           0.4, 0.06, 0.4, 2.50e-3, 0.8e-3, 40])
    array([0.6, 0.75, 250, 203,
           3.8, 3.8, 5.50e-3, 7.8e-3, ])

    bounds = array([(0.5, 1.0), (0.5, 1.0), (100, 400), (100, 400),
                (2.0, 5.0), (2.0, 5.0), (100, 400), (100, 400)])

    p0 = array([1.5, 2.9, 1.5, 0.00, 0.05, 0.00])

    d0 = array([44.0, 650.0, 71.0])

    #optimize_thicknesses(d0, p0, freq_idx_range, bounds)
    optimize_sm(d0, p0, freq_idx_range, bounds)
    plt.show()
    cost_inst = Cost(p0=p0, d_lst=d0, freq_idx_range=freq_idx_range)

    m = freq_idx_range[1] - freq_idx_range[0]  # freq_cnt
    f_opt_amp, f_opt_phi, n_res = np.zeros(m), np.zeros(m), np.zeros((m, len(bounds)))
    for loop_idx, freq_idx in enumerate(range(*freq_idx_range)):
        #cost = partial(cost_inst.cost, freq_idx=freq_idx)
        cost = cost_inst.cost_sm

        res = shgo(cost, bounds=bounds, iters=1)
        # res = basinhopping(cost, x0=p0)

        print(f"Freq: {cost_inst.freqs[freq_idx]} (Idx: {freq_idx}), res.x: {res.x}, res.fun: {res.fun}")
        n_res[loop_idx] = res.x
        amp_loss, phi_loss = cost(res.x, return_both=True)
        f_opt_amp[loop_idx] = amp_loss
        f_opt_phi[loop_idx] = phi_loss

    cost_inst.plot_model(p=n_res)

    f_opt = f_opt_amp + f_opt_phi
    avg_min_cost = np.mean(f_opt)
    print(f"avg_min_cost: {avg_min_cost}")
    print(cost_inst.gof(p=n_res))

    cost_inst.plot_n(p=n_res)

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