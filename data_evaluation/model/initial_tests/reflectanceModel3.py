import numpy as np
from functions import mult_2x2_matrix_chain
from consts import ROOT_DIR, THz, c0, GHz, um, um_to_m
from numpy import cos, sin, exp, array, arcsin, pi, conj, sum, dot
import matplotlib.pyplot as plt
import pandas as pd


def refractive_index(freq_axis):
    """
    load values from files and interpolate according to freq_axis
    unit of freq axis in files is THz
    """
    scale = (100 * c0 / (4 * pi * freq_axis))  # should be in SI

    freq_axis = freq_axis / THz

    mat_data_dir = ROOT_DIR / "model" / "material_data"

    pom_n_path, pom_alpha_path = mat_data_dir / "pom_n.csv", mat_data_dir / "pom_alpha.csv"
    ptfe_n_path, ptfe_alpha_path = mat_data_dir / "ptfe_n.csv", mat_data_dir / "ptfe_alpha.csv"

    pom_n_data, pom_alpha_data = pd.read_csv(pom_n_path), pd.read_csv(pom_alpha_path)
    ptfe_n_data, ptfe_alpha_data = pd.read_csv(ptfe_n_path), pd.read_csv(ptfe_alpha_path)

    pom_n_real = np.interp(freq_axis, pom_n_data.values[:, 0], pom_n_data.values[:, 1])
    pom_n_imag = np.interp(freq_axis, pom_alpha_data.values[:, 0], pom_alpha_data.values[:, 1]) * scale
    pom_n = pom_n_real + 1j * pom_n_imag

    ptfe_n_real = np.interp(freq_axis, ptfe_n_data.values[:, 0], ptfe_n_data.values[:, 1])
    ptfe_n_imag = np.interp(freq_axis, ptfe_alpha_data.values[:, 0], ptfe_alpha_data.values[:, 1]) * scale
    ptfe_n = ptfe_n_real + 1j * ptfe_n_imag

    # debugging. Get same result as refl. model1.py
    #pom_n, ptfe_n = 1.50 * np.ones_like(pom_n), 3.00 * np.ones_like(ptfe_n)

    return pom_n, ptfe_n


class PaperReflectanceModel:
    """
    Approximations used in the analysis of signals inpump-probe spectroscopyPETERKARLSEN1,
    *ANDEUANHENDRY11School of Physics, University of Exeter, Stocker Road, EX4 4QL, United Kingdom*
    Corresponding author: Peterkarlsen88@gmail.com Compiled February 26, 2019
    Research ArticleJournal of the Optical Society of America B

    Layer 0 and N+1 are semi-infinite. (Inbetween are actual layers, #N)
    N = 3; -> 0, 1, 2, 3, 4

    """

    def __init__(self, freq_axis):
        self.freqs = freq_axis
        self.lam = c0 / freq_axis

        self.polarization = "p"#"p" # p-polarization is used in multir, or 'a' switch == 1. Else s-pol.
        self.theta0 = 8 * pi / 180
        self.N = 3
        self.one = np.ones_like(freq_axis)
        pom_n, ptfe_n = refractive_index(freq_axis)
        #n = [ptfe_n, pom_n, 1.1*np.ones_like(freq_axis)]
        n = [1.5 * self.one, 2.8 * self.one, 1.5 * self.one]

        # self.n should have shape (self.N + 2, len(freq_axis))
        self.n = np.array([np.ones_like(freq_axis), *n, np.ones_like(freq_axis)])
        self.theta = np.zeros((self.N + 2, len(freq_axis)))  # last layer doesn't have an interface
        self.delta = np.zeros((self.N + 2, len(freq_axis)), np.complex128)  # delta[0] and delta[N+1] are 0.
        self.gamma = np.zeros((self.N + 2, len(freq_axis)), np.complex128)  # delta[0] and delta[N+1] are 0.
        self.M = np.zeros((self.N, 2, 2, len(freq_axis)), np.complex128)
        self.M_tot = np.zeros((2, 2, len(freq_axis)), np.complex128)
        self.rs, self.ts, self.rp, self.tp = np.zeros((4, self.N + 2, len(freq_axis)), np.complex128)

    def calc_theta(self):
        n = self.n
        self.theta[0] = self.theta0

        for i in range(1, self.N + 2):
            self.theta[i] = arcsin(sin(self.theta[i - 1]) * n[i - 1].real / n[i].real)

    def calc_delta(self, p):
        freqs = self.freqs
        theta = self.theta
        n = self.n
        omega = 2 * pi * freqs

        p = np.array([0, *p.copy(), 0])  # first and last layer are semi-infinite
        for i in range(0, self.N + 2):
            #self.delta[i] = omega * n[i] * p[i] * cos(theta[i]) / c0
            self.delta[i] = omega * n[i] * p[i] / (c0 * cos(theta[i]))

    def calc_fresnel(self):
        t = self.theta
        n = self.n
        for i in range(0, self.N + 1):
            denum_s = n[i] * cos(t[i]) + n[i + 1] * cos(t[i + 1])
            denum_p = n[i + 1] * cos(t[i]) + n[i] * cos(t[i + 1])
            enum_rs = n[i] * cos(t[i]) - n[i + 1] * cos(t[i + 1])
            enum_rp = n[i + 1] * cos(t[i]) - n[i] * cos(t[i + 1])
            enum_t = 2 * n[i] * cos(t[i])

            self.rs[i] = enum_rs / denum_s
            self.ts[i] = enum_t / denum_s

            self.rp[i] = enum_rp / denum_p
            self.tp[i] = enum_t / denum_p

    def calc_M(self):
        delta = self.delta

        if self.polarization == "s":
            r, t = self.rs, self.ts
        else:
            r, t = self.rp, self.tp

        for i in range(1, self.N + 1):
            m00, m01 = exp(-1j * delta[i]), exp(-1j * delta[i]) * r[i]
            m10, m11 = exp(1j * delta[i]) * r[i], exp(1j * delta[i])

            self.M[i - 1] = (1 / t[i]) * np.array([[m00, m01], [m10, m11]])

        M = mult_2x2_matrix_chain(self.M)


        self.M_tot = (1 / t[0]) * mult_2x2_matrix_chain([np.array([[self.one, r[0]], [r[0], self.one]]), M])


    def calc_r(self, p):
        self.calc_theta()
        self.calc_delta(p)
        self.calc_fresnel()
        self.calc_M()

        m00, m10 = self.M_tot[0, 0, :], self.M_tot[1, 0, :]
        #print(m00[0], m10[0])
        r = m10 / m00

        return r


if __name__ == '__main__':
    from consts import custom_mask_420, full_range_mask
    from functions import get_full_measurement

    #model stuff
    # x_axis = np.arange(300, 1500, 0.001).astype(np.float64)
    x_axis = np.arange(1, 10000, 0.01).astype(np.float64)
    freqs = x_axis * GHz

    new_r_model = PaperReflectanceModel(freq_axis=freqs)

    p0 = array([45.0, 620.0, 45.0]) * um_to_m
    #p0 = array([2000.0, 8500.0, 0.0]) * um_to_m

    # p0 = array([2860.0, 0.0, 0.0]) * um_to_m
    refl = new_r_model.calc_r(p0)

    # measurement
    f_z_s, r_z_s, b_z_s, s_z_s = get_full_measurement(sample_file_idx=56)
    idx = (200 * GHz < f_z_s) * (f_z_s < 2600 * GHz)
    t_func_s = (s_z_s - b_z_s) / (r_z_s - b_z_s)
    f_z_s, t_func_s = f_z_s[idx], t_func_s[idx]

    plt.figure()
    plt.plot(f_z_s, np.abs(t_func_s), label="measurement")
    plt.plot(freqs, np.abs(refl), label="model")

    #refl -= np.mean(np.abs(refl))

    refl_fd0 = np.fft.fft(refl, len(refl))
    plt.legend()
    plt.show()

    plt.figure()
    df = np.mean(np.diff(freqs))
    dt = 1 / df

    t = x_axis * dt
    f = np.fft.fftfreq(len(t), float(df))
    idx = f > 0

    plt.plot(f[idx], (np.abs(refl_fd0))[idx], label=f"refl_fd0")
    plt.legend()
    plt.xlabel("time (s)")
    plt.show()

    """
    f_z_s, r_z_s, b_z_s, s_z_s = get_full_measurement(sample_file_idx=56)

    idx = (200 * GHz < f_z_s) * (f_z_s < 600 * GHz)
    t_func_s = (s_z_s - b_z_s) / (r_z_s - b_z_s)

    f_z_s, t_func_s = f_z_s[idx], t_func_s[idx]

    t_func_avg = np.zeros_like(r_z_s)
    for i in range(100):
        f_z, r_z, b_z, s_z = get_full_measurement(sample_file_idx=i)
        idx = (200 * GHz < f_z) * (f_z < 600 * GHz)
        t_func_avg += (s_z - b_z) / (r_z - b_z)

    t_func_avg /= 100
    idx = (200 * GHz < f_z) * (f_z < 600 * GHz)

    f_z, t_func_avg = f_z[idx], t_func_avg[idx]

    plt.figure()
    plt.plot(f_z_s / GHz, np.abs(t_func_s)**2, label="single")
    plt.plot(f_z / GHz, np.abs(t_func_avg) ** 2, label="avg")
    plt.legend()
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("abs(trans func)**2")

    f = np.fft.fftfreq(len(f_z_s), GHz)
    #idx = (f < 0) + (f > 0)

    t_func_fd_s = np.fft.fft(t_func_s, len(f))
    t_func_fd_avg = np.fft.fft(t_func_avg, len(f))

    plt.figure()
    plt.plot(np.abs(t_func_fd_s)[:], label=f"t_func_fd_s")
    plt.plot(np.abs(t_func_fd_avg)[:], label=f"t_func_fd_avg")
    plt.legend()
    #plt.xlabel("time (s)")

    plt.show()
    """