import numpy as np
from consts import um_to_m, THz, GHz, um, array
from model.tmm_reduced import get_amplitude, get_phase, thickest_layer_approximation, get_r_cart, tmm_matrix_elems
from model.initial_tests.explicitEvalSimple import explicit_reflectance_complex
from model.refractive_index import get_n
from optimization.nelder_mead_nD import Point
import matplotlib.pyplot as plt
from functions import noise_gen


class Cost:
    def __init__(self, freqs, p_solution, noise_std_scale=1):
        self.freqs = freqs
        self.n = get_n(freqs, n_min=2.8, n_max=2.8)
        self.en_noise = True
        noise_amp = noise_gen(self.freqs, self.en_noise, scale=0.15 * noise_std_scale, seed=None)
        noise_phase = noise_gen(self.freqs, self.en_noise, scale=0.10 * noise_std_scale, seed=None)

        self.R0_amplitude = get_amplitude(self.freqs, p_solution * um_to_m, self.n) * (1 + noise_amp) ** 2
        self.R0_phase = get_phase(self.freqs, p_solution * um_to_m, self.n) + noise_phase

    def cost(self, point, *args):
        def cost_function(p):
            r_exp = np.sqrt(self.R0_amplitude) * np.exp(1j * self.R0_phase)

            # amp loss only
            """
            amp_loss = sum((get_amplitude(self.freqs, p, self.n) - self.R0_amplitude) ** 2)
            phase_loss = sum((get_phase(self.freqs, p, self.n) - self.R0_phase) ** 2)

            loss = amp_loss

            """

            # cartesian loss
            """
            r_mod = get_r_cart(self.freqs, p, self.n)
            r_exp = np.sqrt(self.R0_amplitude) * np.exp(1j * self.R0_phase)
            #print(r_mod)
            amp_loss = sum((r_mod.real - r_exp.real) ** 2)
            phase_loss = sum((r_mod.imag - r_exp.imag) ** 2)
            """
            # """
            # no_div_loss
            #m01, m11 = tmm_matrix_elems(self.freqs, p, self.n)
            m01, m11 = explicit_reflectance_complex(p)
            # r = tmm_matrix_elems(self.freqs, p, self.n)
            m01_r, m11_r, m01_i, m11_i = m01.real, m11.real, m01.imag, m11.imag
            r_mod_enum_r = (m01_r * m11_r + m01_i * m11_i)  # / (m11_r**2 + m11_i**2)
            r_mod_enum_i = (m01_i * m11_r - m01_r * m11_i)  # / (m11_r**2 + m11_i**2)
            r_mod_denum = m11_r**2 + m11_i**2
            """
            print("r_mod_enum_r", r_mod_enum_r)
            print("r_mod_enum_i", r_mod_enum_i)
            print("r_mod_denum", r_mod_denum)
            print("r_exp.real", r_exp.real)
            print("r_exp.imag", r_exp.imag)
            print("r_exp.real * r_mod_denum", r_exp.real * r_mod_denum)
            print("r_exp.imag * r_mod_denum", r_exp.imag * r_mod_denum)
            print("diff_sqr_real", (r_mod_enum_r - r_exp.real * r_mod_denum) ** 2)
            print("diff_sqr_imag", (r_mod_enum_i - r_exp.imag * r_mod_denum) ** 2, "\n")
            print(r_exp)
            """
            amp_loss = sum((r_mod_enum_r - r_exp.real * r_mod_denum) ** 2)
            phase_loss = sum((r_mod_enum_i - r_exp.imag * r_mod_denum) ** 2)
            """
            s = 0
            for i in range(6):
                s += (r_mod_enum_r[i] - r_exp.real[i] * r_mod_denum[i]) ** 2 + \
                     (r_mod_enum_i[i] - r_exp.imag[i] * r_mod_denum[i]) ** 2
                print(s)
            """

            """
            r_mod = get_r_cart(self.freqs, p, self.n)
            r_exp = np.sqrt(self.R0_amplitude) * np.exp(1j*self.R0_phase)

            amp_loss = sum((r_mod.real - r_exp.real) ** 2)
            phase_loss = sum((r_mod.imag - r_exp.imag) ** 2)

            loss = np.log10(amp_loss * phase_loss)
            """
            loss = amp_loss + phase_loss

            return loss

        if type(point) is np.ndarray:
            p = point.copy() * um_to_m

            return cost_function(p)
        else:
            p = array([point.x[0], point.x[1], point.x[2]], dtype=float) * um_to_m

            point.fx = cost_function(p)


if __name__ == '__main__':
    freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950]) * THz  # GHz; freqs. set on fpga
    p_sol = array([193.0, 544.0, 168.0])
    #p_sol = array([293.0, 344.0, 108.0])
    #p_sol = array([50.0, 400.0, 50.0])
    # for _ in range(100):
    new_cost = Cost(freqs, p_sol, noise_std_scale=0)
    cost_func = new_cost.cost
    # cost_func(p_sol)
    p = array([150.0, 400.0, 50.0])
    p = array([50, 450, 100])
    #p = array([50, 450, 50])

    print(cost_func(p))
    exit()

    """ # noise can make fx of p_sol (also small variations of p_sol?) higher than other candidates. 
    p_test = array([ 79.07, 530.87, 334.56])
    print("truth fx: ", cost_func(p_sol))
    print("found fx: ", cost_func(p_test))
    print(cost_func(p_sol) < cost_func(p_test))
    """
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
