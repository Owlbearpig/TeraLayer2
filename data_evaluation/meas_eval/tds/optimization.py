import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize, shgo, basinhopping
from functools import partial
from cost import Cost
from consts import *


def optimize_thicknesses(d0, freq_idx_range, bounds):
    best_fit_pcc, max_val_pcc = None, 0
    best_fit_loss, min_val_loss = None, np.inf
    m = freq_idx_range[1] - freq_idx_range[0]  # freq_cnt
    for x in range(-5, 6):
        for y in range(-5, 6):
            z = -5
            print(x, y, z)
            d_lst = [d0[0] + x, d0[1] + y, d0[2] + z]
            cost_inst = Cost(d_lst=d_lst, freq_idx_range=freq_idx_range)

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

        print(f"best_fit_pcc: {best_fit_pcc}, max_val_pcc: {max_val_pcc}")
        print(f"best_fit_loss: {best_fit_loss}, min_val_loss: {min_val_loss}")

        return best_fit_pcc, best_fit_loss


def optimize_sell(d0, freq_idx_range, bounds):
    cost_inst = Cost(d_lst=d0, freq_idx_range=freq_idx_range, model="sell")

    res = shgo(cost_inst.cost, bounds=bounds)
    print(res)
    print(cost_inst.gof(p=res.x))

    cost_inst.plot_model(p=res.x)
    cost_inst.plot_n(p=res.x)


def optimize_direct(d0, freq_idx_range, bounds, plot_fun=False):
    cost_inst = Cost(d_lst=d0, freq_idx_range=freq_idx_range)

    m = freq_idx_range[1] - freq_idx_range[0]  # freq_cnt
    f_opt_amp, f_opt_phi, n_res = np.zeros(m), np.zeros(m), np.zeros((m, len(bounds)))
    for loop_idx, freq_idx in enumerate(range(*freq_idx_range)):
        cost = partial(cost_inst.cost, freq_idx=freq_idx)

        res = shgo(cost, bounds=bounds, iters=2)
        #print(res)
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

    if plot_fun:
        plt.figure("Fun val")
        plt.plot(cost_inst.freq_range, np.log10(f_opt), label="f_opt")
        plt.plot(cost_inst.freq_range, np.log10(f_opt_amp), label="f_opt_amp")
        plt.plot(cost_inst.freq_range, np.log10(f_opt_phi), label="f_opt_phi")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("log10(val)")
        plt.legend()

    return n_res


def main():
    df = 0.014275517487508922
    # f0_idx = int(0.150 / df)
    # f1_idx = int(4.000 / df)
    f0_idx = int(0.350 / df)
    f1_idx = int(3.500 / df)

    freq_idx_range = f0_idx, f1_idx

    d0 = array([44.0, 650.0, 71.0])
    bounds_reg = array([(1.49, 1.51), (2.86, 2.90), (1.49, 1.51), (0.00, 0.035)])
    bounds_reg = array([(1.45, 1.55), (2.70, 3.00), (1.45, 1.55)])
    bounds_reg = array([(2.70, 3.00)])
    #bounds_sell = array([(0.8, 1.4), (200, 600), (5, 8), (200, 500)])
    #bounds_simple = array([(1.1, 2.0), (3.5, 4.0), (1.1, 2.0)])

    # optimize_thicknesses(d0, freq_idx_range, bounds_reg)
    # optimize_sell(d0, freq_idx_range, bounds_simple)
    optimize_direct(d0, freq_idx_range, bounds_reg)

    plt.show()


if __name__ == '__main__':
    main()
    plt.show()
