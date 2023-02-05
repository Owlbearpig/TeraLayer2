import matplotlib.pyplot as plt

from scipy.optimize import minimize
from functools import partial
from cost import Cost
from consts import *

def main():
    from scipy.optimize import minimize
    from functools import partial

    cost_inst = Cost()
    f0_idx = int(0.300 / 0.014275517487508922)
    f1_idx = int(2.000 / 0.014275517487508922)
    freq_idx_range = f0_idx, f1_idx

    bounds = [(1.45, 1.55), (2.75, 2.85), (1.45, 1.55)]
    p0 = array([1.5, 2.8, 1.5])

    n_opt = []
    for freq_idx in range(*freq_idx_range):
        cost = partial(cost_inst.cost, freq_idx=freq_idx)
        res = minimize(cost, x0=p0, bounds=bounds)

        print(cost_inst.freqs[freq_idx], res.x)
        n_opt.append(res.x)

    cost_inst.plot_model(p=n_opt, freq_idx_range=freq_idx_range)


if __name__ == '__main__':
    main()
    plt.show()