import numpy as np
from consts import um_to_m, THz, GHz, um, array
from model.tmm_reduced import get_amplitude, get_phase, thickest_layer_approximation, get_r_cart, tmm_matrix_elems
from model.initial_tests.explicitEvalSimple import explicit_reflectance_complex
from model.refractive_index import get_n
from optimization.nelder_mead_nD import Point
import matplotlib.pyplot as plt
from functions import noise_gen
from functools import partial


class Cost:
    def __init__(self, freqs=None, p_solution=array([200, 600, 300]), noise_std_scale=0.0, seed=420, plt_mod=False):
        if freqs is None:
            self.freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950]) * THz  # GHz; freqs. set on fpga
        else:
            self.freqs = freqs

        self.p_solution = array(p_solution)

        self.all_freqs = np.arange(0.000, 1500 + 1, 1) * GHz

        self.n = get_n(self.freqs, n_min=2.8, n_max=2.8)

        approximate_model = False
        if not approximate_model:
            self.R0_amplitude = get_amplitude(self.freqs, self.p_solution * um_to_m, self.n)
            self.R0_phase = get_phase(self.freqs, self.p_solution * um_to_m, self.n)
        else:
            m01, m11 = explicit_reflectance_complex(self.p_solution * um_to_m)
            r = m01 / m11
            self.R0_amplitude = np.real(r * np.conj(r))
            self.R0_phase = np.angle(r)

        self.noise_std_scale = noise_std_scale
        self.noise_amp = noise_gen(self.all_freqs, True, scale=0.15 * noise_std_scale, seed=seed)
        self.noise_phase = noise_gen(self.all_freqs, True, scale=0.20 * noise_std_scale, seed=seed)

        selected_freqs_idx = array([np.argwhere(np.isclose(freq, self.all_freqs))[0][0] for freq in self.freqs])

        self.R0_amplitude *= self.noise_amp[selected_freqs_idx] ** 2
        self.R0_phase += (1 - self.noise_phase[selected_freqs_idx])

        self.r_exp = np.sqrt(self.R0_amplitude) * np.exp(1j * self.R0_phase)

        if plt_mod:
            self.plot_model()
            plt.show()

    def cost(self, point, *args):
        def cost_function(p):
            # p should be in meter (m) -> e.g. 420 * 10^-6 m if 420 um
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
            r_mod_denum = m11_r ** 2 + m11_i ** 2
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
            #"""
            #print("r_mod_denum: ", r_mod_denum)
            #print("r_mod_enum_r / r_mod_denum: ", r_mod_enum_r / r_mod_denum, "r_exp.real: ", self.r_exp.real)
            #print("r_mod_enum_i / r_mod_denum: ", r_mod_enum_i / r_mod_denum, "self.r_exp.imag: ", self.r_exp.imag)

            amp_loss = sum((r_mod_enum_r  - self.r_exp.real * r_mod_denum) ** 2)
            phase_loss = sum((r_mod_enum_i  - self.r_exp.imag * r_mod_denum) ** 2)
            #"""
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

            return loss / 2

        if type(point) is np.ndarray:
            if all([point[i] < 0.1 for i in range(3)]):
                p = point.copy()
            else:
                p = point.copy() * um_to_m

            return cost_function(p)
        else:
            if all([point.x[i] < 0.1 for i in range(3)]):
                p = array([point.x[0], point.x[1], point.x[2]], dtype=float)
            else:
                p = array([point.x[0], point.x[1], point.x[2]], dtype=float) * um_to_m

            point.fx = cost_function(p)

    def plot_model(self, ):
        m01, m11 = explicit_reflectance_complex(freqs=self.all_freqs, p=self.p_solution * um_to_m)
        r = m01 / m11
        R0_amplitude = np.real(r * np.conj(r))
        R0_amplitude_noisy = R0_amplitude * self.noise_amp ** 2

        R0_phase = np.angle(r)
        R0_phase_noisy = R0_phase + (1 - self.noise_phase)

        r_noisy = np.sqrt(R0_amplitude) * np.exp(1j * R0_phase_noisy)

        fig, (ax01, ax02) = plt.subplots(1, 2)
        ax01.plot(self.all_freqs / GHz, R0_phase, label=f"Phase (no noise), p_sol={self.p_solution}")
        ax01.plot(self.all_freqs / GHz, R0_phase_noisy, label=f"Phase noisy scale={self.noise_std_scale}")
        ax01.set_xlabel("Frequency (GHz)")
        ax01.set_ylabel("Phase (rad)")
        ax01.legend()

        ax02.plot(self.all_freqs / GHz, R0_amplitude, label=f"Intensity (no noise), p_sol={self.p_solution}")
        ax02.plot(self.all_freqs / GHz, R0_amplitude_noisy, label=f"Intensity noisy scale={self.noise_std_scale}")
        ax02.set_xlabel("Frequency (GHz)")
        ax02.set_ylabel("Int. (a.u.)")
        ax02.legend()

        fig, (ax11, ax12) = plt.subplots(1, 2)
        ax11.plot(self.all_freqs / GHz, np.imag(r), label=f"Im(r) (no noise), p_sol={self.p_solution}")
        ax11.plot(self.all_freqs / GHz, np.imag(r_noisy), label=f"Im(r) noisy scale={self.noise_std_scale}")
        ax11.set_xlabel("Frequency (GHz)")
        ax11.set_ylabel("Im(Reflectance)")
        ax11.legend()

        ax12.plot(self.all_freqs / GHz, np.real(r), label=f"Re(r) (no noise), p_sol={self.p_solution}")
        ax12.plot(self.all_freqs / GHz, np.real(r_noisy), label=f"Re(r) noisy scale={self.noise_std_scale}")
        ax12.set_xlabel("Frequency (GHz)")
        ax12.set_ylabel("Re(Reflectance)")
        ax12.legend()

        selected_freqs = self.freqs / GHz
        for xc in selected_freqs:
            ax01.axvline(x=xc, color="red")
            ax02.axvline(x=xc, color="red")
            ax11.axvline(x=xc, color="red")
            ax12.axvline(x=xc, color="red")


if __name__ == '__main__':
    #freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950]) * THz  # GHz; freqs. set on fpga
    #freqs = np.arange(0.000, 1.500 + 0.001, 0.001) * THz
    freqs = None
    #p_sol = array([193.0, 544.0, 168.0])
    #p_sol = array([170, 690, 69])
    #p_sol = array([168., 609.,  98.])
    #p_sol = array([293.0, 344.0, 108.0])
    #p_sol = array([50.0, 400.0, 50.0])
    p_sol = array([42.0, 641.0, 74.0])
    noise_factor = 0.0
    #p_sol = array([290.0, 658.0, 94.0])
    new_cost = Cost(freqs=freqs, p_solution=p_sol, noise_std_scale=noise_factor, plt_mod=True)
    cost_func = new_cost.cost
    # cost_func(p_sol)
    #p = array([150.0, 500.0, 100.0])
    #p = array([ 260., 651.,  50.])
    p = array([239.777814149857, 476.259423971176, 235.382882833481])
    #p = array([299.0, 603.0, 71.0])
    #p = array([150., 500., 150.])
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
