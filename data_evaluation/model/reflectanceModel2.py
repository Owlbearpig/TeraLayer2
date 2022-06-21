import numpy as np
from functions import mult_2x2_matrix_chain
from consts import ROOT_DIR, THz, c0, GHz, um, um_to_m
from numpy import cos, sin, exp, array, arcsin, pi, conj, sum
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

    pom_n, ptfe_n = 1.68*np.ones_like(pom_n), 1.35*np.ones_like(ptfe_n) # debugging. Get same result as refl. model1.py

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

        self.polarization = "TE"
        self.theta0 = 10 * pi / 180
        self.N = 3
        pom_n, ptfe_n = refractive_index(freq_axis)
        n = [ptfe_n, pom_n, np.ones_like(freq_axis)]

        # self.n should have shape (self.N + 2, len(freq_axis))
        self.n = np.array([np.ones_like(freq_axis), *n, np.ones_like(freq_axis)])
        self.theta = np.zeros((self.N + 2, len(freq_axis)))  # last layer doesn't have an interface
        self.delta = np.zeros((self.N + 2, len(freq_axis)), np.complex128)  # delta[0] and delta[N+1] are 0.
        self.gamma = np.zeros((self.N + 2, len(freq_axis)), np.complex128)  # delta[0] and delta[N+1] are 0.
        self.M = np.zeros((self.N, 2, 2, len(freq_axis)), np.complex128)
        self.M_tot = np.zeros((2, 2, len(freq_axis)), np.complex128)

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
            self.delta[i] = omega * n[i] * p[i] * cos(theta[i]) / c0

    def calc_gamma(self):
        n = self.n

        for i in range(0, self.N + 2):
            if self.polarization == "TE":
                self.gamma[i] = cos(self.theta[i]) * n[i]
            else:
                self.gamma[i] = cos(self.theta[i]) / n[i]

    def calc_M(self):
        delta, gamma = self.delta, self.gamma

        for i in range(1, self.N + 1):
            m00, m01 = cos(delta[i]), -1j * sin(delta[i]) / gamma[i]
            m10, m11 = -1j * gamma[i] * sin(delta[i]), cos(delta[i])

            self.M[i - 1] = np.array([[m00, m01],
                                      [m10, m11]])

        self.M_tot = mult_2x2_matrix_chain(self.M, self.N)

    def calc_r(self, p):
        self.calc_theta()
        self.calc_delta(p)
        self.calc_gamma()
        self.calc_M()

        m00, m01 = self.M_tot[0, 0, :], self.M_tot[0, 1, :]
        m10, m11 = self.M_tot[1, 0, :], self.M_tot[1, 1, :]

        g = self.gamma

        enum_ = g[0] * m00 + g[0] * g[-1] * m01 - m10 - g[-1] * m11
        denum_ = g[0] * m00 + g[0] * g[-1] * m01 + m10 + g[-1] * m11

        return enum_ / denum_


if __name__ == '__main__':
    from consts import custom_mask_420, full_range_mask
    # x_axis = np.arange(300, 1500, 0.001).astype(np.float64)
    x_axis = np.arange(200, 1500, 1).astype(np.float64)
    freqs = x_axis * GHz

    new_r_model = PaperReflectanceModel(freq_axis=freqs)
    #new_r_model.polarization = "TE"

    p0 = array([2860.0, 4997.0, 0.0]) * um_to_m
    p0 = array([2860.0, 0.0, 0.0]) * um_to_m
    r = new_r_model.calc_r(p0)

    plt.plot(freqs, (np.abs(r)))
    plt.show()
