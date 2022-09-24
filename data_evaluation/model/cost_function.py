import numpy as np
from consts import um_to_m, THz, GHz, um, array
from model.tmm import get_amplitude, get_phase, thickest_layer_approximation
from model.refractive_index import get_n
from optimization.nelder_mead_nD import Point
import matplotlib.pyplot as plt
from functions import noise_gen


class Cost:
    def __init__(self, freqs, p_solution):
        self.freqs = freqs
        self.n = get_n(freqs, n_min=2.8, n_max=2.8)
        self.en_noise = True
        noise_amp = noise_gen(self.freqs, self.en_noise, scale=0.08, seed=420)
        noise_phase = noise_gen(self.freqs, self.en_noise, scale=0.15, seed=421)

        self.R0_amplitude = get_amplitude(self.freqs, p_solution * um_to_m, self.n) + noise_amp
        self.R0_phase = get_phase(freqs, p_solution * um_to_m, self.n) + noise_phase

    def cost(self, point, *args):
        def cost_function(p):
            amp_loss = sum((get_amplitude(self.freqs, p, self.n) - self.R0_amplitude) ** 2)
            phase_loss = sum((get_phase(self.freqs, p, self.n) - self.R0_phase) ** 2)

            loss = np.log10(amp_loss * phase_loss)

            return loss

        if isinstance(point, Point):
            p = array([point.x[0], point.x[1], point.x[2]], dtype=float) * um_to_m

            point.fx = cost_function(p)
        else:
            p = point.copy() * um_to_m

            return cost_function(p)


if __name__ == '__main__':
    freqs = array([0.040, 0.080, 0.150, 0.550, 0.640, 0.760]) * THz  # pretty good
    p_sol = array([118.0, 513.0, 206.0])

    new_cost = Cost(freqs, p_sol)
    cost_func = new_cost.cost

    rez = 1
    x = np.arange(0, 1000, rez)
    y1 = array([cost_func(array([d1, p_sol[1], p_sol[2]])) for d1 in x])
    y2 = array([cost_func(array([p_sol[0], d2, p_sol[2]])) for d2 in x])
    y3 = array([cost_func(array([p_sol[0], p_sol[1], d3])) for d3 in x])
    print(np.argmin(y1) * rez, np.argmin(y2) * rez, np.argmin(y3) * rez)
    plt.title(f"Loss function with solution at {p_sol}")
    plt.plot(x, y1, label=f"d1, [d, {p_sol[1]}, {p_sol[2]}]")
    plt.plot(x, y2, label=f"d2, [{p_sol[0]}, d, {p_sol[2]}]")
    plt.plot(x, y3, label=f"d3, [{p_sol[0]}, {p_sol[1]}, d]")
    plt.xticks(np.arange(0, 1000 + 100, 100))
    plt.xlabel("d ($\mu$m)")
    plt.ylabel("$log_{10}$(AmpLoss * PhaseLoss)")
    plt.legend()
    plt.show()
