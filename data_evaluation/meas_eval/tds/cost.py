import numpy as np
from scipy.stats import pearsonr
from main import load_data
from functions import filtering, do_fft, do_ifft
from functools import partial
from visualizing.simplecolormap import map_plot
from model.refractive_index import get_n
import matplotlib.pyplot as plt
from mpl_settings import *
from model.tmm_package import tmm_package_wrapper
from consts import *


class Cost:
    def __init__(self, sam_idx=None, n_pad=None, d_lst=None, freq_idx_range=None, model=None):
        self.sam_idx = sam_idx
        self.model = model

        if d_lst is None:
            self.d_list = array([42.0, 641.0, 74.0])
        else:
            self.d_list = array(d_lst)

        if n_pad is None:
            self.n_pad = array([1.5, 2.9, 1.5])
        else:
            self.n_pad = array(n_pad)[:3]

        self.ref_td, self.sam_td = load_data(sam_idx=self.sam_idx, signal_shift=-5)

        en_filter = False
        if en_filter:
            self.ref_td = filtering(self.ref_td, filt_type="bp", wn=(1.0, 2.3))
            self.sam_td = filtering(self.sam_td, filt_type="bp", wn=(1.0, 2.3))

        self.sam_fd, self.ref_fd = do_fft(self.sam_td), do_fft(self.ref_td)

        self.freqs = self.ref_fd[:, 0].real
        self.lam = c_thz / self.freqs
        self.t = self.ref_td[:, 0].real

        self.r_exp = array([self.freqs, self.sam_fd[:, 1] / self.ref_fd[:, 1]]).T

        self.k = np.array([np.linspace(0, 0.00, len(self.freqs)),
                           np.linspace(0, 0.10, len(self.freqs)),
                           np.linspace(0, 0.00, len(self.freqs))]).T

        if freq_idx_range is None:
            self.freq_idx_range = (0, len(self.freqs))
        else:
            self.freq_idx_range = freq_idx_range

        self.freq_range = self.freqs[self.freq_idx_range[0]:self.freq_idx_range[1]]

    def cost(self, p, return_both=False, **kwargs):
        p = array(p)
        kwargs["return_both"] = return_both
        if self.model == "sell":
            return self.cost_sm(p, **kwargs)
        else:
            return self.cost_reg(p, **kwargs)

    def cost_reg(self, p, **kwargs):
        freq_idx = kwargs["freq_idx"]
        return_both = kwargs["return_both"]

        if len(p) == 1:
            n = array([1.5, p[0], 1.5]) + 1j * self.k[freq_idx]
        elif len(p) == 2:
            n = array([p[0], p[1], p[0]]) + 1j * self.k[freq_idx]
        elif len(p) == 3:
            n = array(p) + 1j * self.k[freq_idx]
        elif len(p) == 4:
            n = array(p[:3]) + 1j * array(p[3]) * array([0, 1, 0])
        else:
            n = array(p[:3]) + 1j * array(p[3:])

        r_tmm = tmm_package_wrapper(self.freqs[freq_idx], self.d_list, n)
        r_exp = self.sam_fd[freq_idx, 1] / self.ref_fd[freq_idx, 1]

        return self.loss(r_tmm[1], r_exp, return_both)

    def cost_sm(self, p, **kwargs):
        return_both = kwargs["return_both"]
        n = self.n_sm(p)
        f0_i, f1_i = self.freq_idx_range

        r_tmm = tmm_package_wrapper(self.freq_range, self.d_list, n[f0_i:f1_i])
        r_exp = self.r_exp[f0_i:f1_i, 1]

        return self.loss(r_tmm[:, 1], r_exp, return_both)

    def loss(self, r_model, r_exp, return_both=False):
        amp_loss = (np.log10(np.abs(r_model)) - np.log10(np.abs(r_exp))) ** 2
        phi_loss = (np.angle(r_model) - np.angle(r_exp)) ** 2

        if return_both:
            return amp_loss, phi_loss
        else:
            loss = amp_loss + phi_loss

            if self.model == "sell":
                return np.sum(loss) / len(loss)
            else:
                return loss

    def calc_n(self, p):
        if self.model == "sell":
            n = self.n_sm(p)
        else:
            n = self.pad_n(p, self.freq_idx_range)

        return n

    def n_sm(self, p):
        l = self.lam ** 2
        """
        B1, B2, C1, C2 = p[:4]
        n1 = np.sqrt(1 + B1 * l / (l - C1) + B2 * l / (l - C2))

        B1, B2, C1, C2 = p[4:8]
        n2 = np.sqrt(1 + B1 * l / (l - C1) + B2 * l / (l - C2))
        """

        B1, C1 = p[:2]
        n1 = np.sqrt(1 + B1 * l / (l - C1))

        B1, C1 = p[2:4]
        n2 = np.sqrt(1 + B1 * l / (l - C1))

        n = array([n1, n2, n1]).T + 1j * self.k

        n = np.nan_to_num(n)

        return n

    def pad_n(self, p, freq_idx_range):
        f0_i, f1_i = freq_idx_range
        pre_pad = np.vstack(f0_i * [self.n_pad])
        post_pad = np.vstack((len(self.freqs) - f1_i) * [self.n_pad])

        p_dim = p.shape[1]

        if p_dim == 1:
            one = np.ones(len(p))
            p = array([self.n_pad[0]*one, p[:, 0], self.n_pad[2]*one]).T
        elif p_dim == 2:
            p = array([p[:, 0], p[:, 1], p[:, 0]]).T

        k = self.k.copy()

        if p_dim == 4:
            k[f0_i:f1_i, 1] = np.array(p[:, 3])
        elif p_dim == 6:
            k[f0_i:f1_i, :] = np.array(p[:, 3:])

        n = np.concatenate((pre_pad, p[:, :3], post_pad)) + 1j * k

        return n

    def calc_model(self, p, ret_td=True):
        n = self.calc_n(p)

        r_tmm_fd = tmm_package_wrapper(self.freqs, self.d_list, n)

        tmm_fd = array([self.freqs, r_tmm_fd[:, 1] * self.ref_fd[:, 1]]).T

        if ret_td:
            tmm_td = do_ifft(tmm_fd)

            return tmm_fd, tmm_td
        else:
            return tmm_fd

    def gof(self, p):
        res = {}
        tmm_fd, tmm_td = self.calc_model(p)
        res["peas_corr_coeff"] = pearsonr(self.sam_td[:, 1].real, tmm_td[:, 1].real)

        return res

    def plot_n(self, p):
        f0_i, f1_i = self.freq_idx_range
        n = self.calc_n(p)

        plt.figure("Refractive index real")
        plt.title(f"{self.d_list}")
        plt.plot(self.freqs, n[:, 0].real, label="Re($n_0$)")
        plt.plot(self.freqs, n[:, 1].real, label="Re($n_1$)")
        plt.plot(self.freqs, n[:, 2].real, label="Re($n_2$)")
        plt.axvline(self.freqs[f0_i], color="red", label="Fit range", linewidth=5.0)
        plt.axvline(self.freqs[f1_i], color="red", linewidth=5.0)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Re(n)")
        plt.legend()

        plt.figure("Refractive index imag")
        plt.title(f"{self.d_list}")
        plt.plot(self.freqs, n[:, 0].imag, label="Img($n_0$)")
        plt.plot(self.freqs, n[:, 1].imag, label="Img($n_1$)")
        plt.plot(self.freqs, n[:, 2].imag, label="Img($n_2$)")
        plt.axvline(self.freqs[f0_i], color="red", label="Fit range", linewidth=5.0)
        plt.axvline(self.freqs[f1_i], color="red", linewidth=5.0)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Im(n)")
        plt.legend()

    def plot_model(self, p):
        f0_i, f1_i = self.freq_idx_range
        tmm_fd, tmm_td = self.calc_model(p)

        plt.figure("Spectrum")
        plt.title(f"{self.d_list}")
        plt.plot(self.freqs, 20 * np.log10(np.abs(tmm_fd[:, 1])), label="r_tmm")
        plt.axvline(self.freqs[f0_i], color="red", label="Fit range", linewidth=5.0)
        plt.axvline(self.freqs[f1_i], color="red", linewidth=5.0)

        plt.figure("Phase")
        plt.title(f"{self.d_list}")
        plt.plot(self.freqs, np.angle(tmm_fd[:, 1] / self.ref_fd[:, 1]), label="tmm")
        plt.axvline(self.freqs[f0_i], color="red", label="Fit range", linewidth=5.0)
        plt.axvline(self.freqs[f1_i], color="red", linewidth=5.0)

        plt.figure("Time domain")
        plt.title(f"{self.d_list}")
        plt.plot(tmm_td[:, 0].real, tmm_td[:, 1], label=f"TMM * Reference")
        self.plot_sam()

    def plot_sam(self):
        plt.figure("Time domain")
        plt.plot(self.t, self.ref_td[:, 1], label=f"Ref. (Meas. idx: {self.sam_idx})")
        plt.plot(self.t, self.sam_td[:, 1], label=f"Sam. (Meas. idx: {self.sam_idx})")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

        plt.figure("Spectrum")
        plt.plot(self.freqs, 20 * np.log10(np.abs(self.ref_fd[:, 1])), label=f"Ref. (Meas. idx: {self.sam_idx})")
        plt.plot(self.freqs, 20 * np.log10(np.abs(self.sam_fd[:, 1])), label=f"Sam. (Meas. idx: {self.sam_idx})")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")
        plt.legend()

        plt.figure("Phase")
        # plt.plot(self.sam_fd[:, 0], np.angle(self.sam_fd[:, 1]), label="$\phi_{sam}$")
        # plt.plot(self.sam_fd[:, 0], np.angle(self.ref_fd[:, 1]), label="$\phi_{ref}$")
        plt.plot(self.sam_fd[:, 0], np.angle(self.r_exp[:, 1]), label=f"(Sam idx: {self.sam_idx}) "
                                                                      + "$\phi_{sam} - \phi_{ref}$")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (Rad)")
        plt.legend(loc='upper right')


def plot_cost(d0, df):
    cost_inst = Cost(d_lst=d0)
    freq = 1.800
    freq_idx = int(freq / df)

    cost = partial(cost_inst.cost, freq_idx=freq_idx)
    print(cost_inst.freqs[freq_idx])
    rez = 100
    one = np.ones(rez)
    n1_arr = np.linspace(1.3, 1.6, rez)
    n2_arr = np.linspace(2.7, 3.1, rez)
    n3_arr = np.linspace(1.3, 1.6, rez)

    lb = array([1.3, 2.7, 1.3])
    ub = array([1.6, 3.1, 1.6])
    settings = {"rez": (100, 100, 100), "ub": ub, "lb": lb, "unit_lbl": ""}
    map_plot(error_func=cost, settings=settings, representation="log")

    ps = np.array([1.5*one, n2_arr, 1.5*one]).T
    vals = []
    for p in ps:
        val = cost(p)
        vals.append(val)

    plt.figure()
    plt.plot(n2, vals)

    plt.figure()
    plt.plot(n2, np.log10(vals))
    plt.show()


def main():
    from functools import partial

    df = 0.014275517487508922
    f0_idx = int(0.150 / df)
    f1_idx = int(4.000 / df)

    d0 = array([44.0, 650.0, 71.0])
    cost_inst = Cost(d_lst=d0, freq_idx_range=(f0_idx, f1_idx), model="sell")

    plot_cost(d0, df)

    # f_idx = int(1.68 / 0.014275517487508922)
    f_idx = 80
    print(f"Freq: {cost_inst.freqs[f_idx]} THz (idx: {f_idx})")

    cost = cost_inst.cost

    p0 = array([1.1, 400, 7.6, 300])

    res = cost(p=p0)
    cost_inst.plot_n(p0)
    cost_inst.plot_model(p0)

    print(res)


if __name__ == '__main__':
    main()
    plt.show()
