import numpy as np
from scipy.stats import pearsonr
from main import load_data
from functions import filtering, do_fft, do_ifft
from model.refractive_index import get_n
import matplotlib.pyplot as plt
from mpl_settings import *
from model.tmm_package import tmm_package_wrapper
from consts import *


class Cost:
    def __init__(self, sam_idx=None, p0=None, d_lst=None, freq_idx_range=None):
        self.sam_idx = sam_idx
        self.p0 = array(p0)

        if d_lst is None:
            self.d_list = array([42.0, 641.0, 74.0])
        else:
            self.d_list = array(d_lst)

        self.ref_td, self.sam_td = load_data(sam_idx=self.sam_idx, signal_shift=-5)

        en_filter = False
        if en_filter:
            self.ref_td = filtering(self.ref_td, filt_type="hp", wn=2.3)
            self.sam_td = filtering(self.sam_td, filt_type="hp", wn=2.3)

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

    def cost(self, p, freq_idx, return_both=False):
        p = array(p)
        if len(p) == 3:
            n = array(p) + 1j * self.k[freq_idx]
        elif len(p) == 4:
            n = array(p[:3]) + 1j * array(p[3]) * array([0, 1, 0])
        else:
            n = array(p[:3]) + 1j * array(p[3:])

        r_tmm = tmm_package_wrapper(self.freqs[freq_idx], self.d_list, n)
        r_tmm[1] = r_tmm[1] * -1
        r_exp = self.sam_fd[freq_idx, 1] / self.ref_fd[freq_idx, 1]

        amp_loss = (np.log10(np.abs(r_exp)) - np.log10(np.abs(r_tmm[1]))) ** 2
        phi_loss = (np.angle(r_tmm[1]) - np.angle(self.r_exp[freq_idx, 1])) ** 2

        loss = amp_loss + phi_loss

        if return_both:
            return amp_loss, phi_loss
        else:
            return loss

    def cost_sm(self, p, return_both=False):
        n = self.n_sm(p)
        f0_i, f1_i = self.freq_idx_range

        r_tmm = tmm_package_wrapper(self.freq_range, self.d_list, n[f0_i:f1_i])
        r_tmm[:, 1] = r_tmm[:, 1] * -1
        r_exp = self.r_exp[f0_i:f1_i, 1]

        amp_loss = (np.log10(np.abs(r_tmm[:, 1])) - np.log10(np.abs(r_exp))) ** 2
        phi_loss = (np.angle(r_tmm[:, 1]) - np.angle(r_exp)) ** 2

        loss = amp_loss + phi_loss

        if return_both:
            return amp_loss, phi_loss
        else:
            return np.sum(loss) / len(loss)

    def n_sm(self, p):
        l = self.lam ** 2

        B1, B2, C1, C2 = p[:4]
        n1 = np.sqrt(1 + B1 * l / (l - C1) + B2 * l / (l - C2))

        B1, B2, C1, C2 = p[4:8]
        n2 = np.sqrt(1 + B1 * l / (l - C1) + B2 * l / (l - C2))

        n = array([n1, n2, n1]).T + 1j * self.k

        n = np.nan_to_num(n)

        return n

    def pad_n(self, p, freq_idx_range):
        if self.p0 is not None:
            pad = self.p0[:3]
        else:
            pad = array([1.5, 2.8, 1.5])

        f0_i, f1_i = freq_idx_range
        pre_pad, post_pad = np.vstack(f0_i * [pad]), np.vstack((len(self.freqs) - f1_i) * [pad])

        k = self.k.copy()
        if p.shape[1] == 4:
            k[f0_i:f1_i, 1] = np.array(p[:, 3])
        elif p.shape[1] == 6:
            k[f0_i:f1_i, :] = np.array(p[:, 3:])

        n = np.concatenate((pre_pad, p[:, :3], post_pad)) + 1j * k

        return n

    def calc_model(self, p, ret_td=True, sm=False):
        if sm:
            n = self.n_sm(p)
        else:
            n = self.pad_n(p, self.freq_idx_range)
        r_tmm_fd = tmm_package_wrapper(self.freqs, self.d_list, n)
        r_tmm_fd[:, 1] *= -1

        tmm_fd = array([self.freqs, r_tmm_fd[:, 1] * self.ref_fd[:, 1]]).T

        if ret_td:
            tmm_td = do_ifft(tmm_fd)

            return tmm_fd, tmm_td
        else:
            return tmm_fd

    def gof(self, p, sm=False):
        res = {}
        tmm_fd, tmm_td = self.calc_model(p, sm=sm)
        res["peas_corr_coeff"] = pearsonr(self.sam_td[:, 1].real, tmm_td[:, 1].real)

        return res

    def plot_n(self, p, sm=False):
        f0_i, f1_i = self.freq_idx_range
        if sm:
            n = self.n_sm(p)
        else:
            n = self.pad_n(p, self.freq_idx_range)

        plt.figure("Refractive index real")
        plt.title(f"{self.d_list}")
        plt.plot(self.freqs, n[:, 0].real, label="Re($n_0$)")
        plt.plot(self.freqs, n[:, 1].real, label="Re($n_1$)")
        plt.plot(self.freqs, n[:, 2].real, label="Re($n_2$)")
        plt.axvline(self.freqs[f0_i], color="red", label="Opt. range", linewidth=5.0)
        plt.axvline(self.freqs[f1_i], color="red", linewidth=5.0)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Re(n)")
        plt.legend()

        plt.figure("Refractive index imag")
        plt.title(f"{self.d_list}")
        plt.plot(self.freqs, n[:, 0].imag, label="Img($n_0$)")
        plt.plot(self.freqs, n[:, 1].imag, label="Img($n_1$)")
        plt.plot(self.freqs, n[:, 2].imag, label="Img($n_2$)")
        plt.axvline(self.freqs[f0_i], color="red", label="Opt. range", linewidth=5.0)
        plt.axvline(self.freqs[f1_i], color="red", linewidth=5.0)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Im(n)")
        plt.legend()

    def plot_model(self, p, sm=False):
        f0_i, f1_i = self.freq_idx_range
        tmm_fd, tmm_td = self.calc_model(p, sm=sm)

        plt.figure("Spectrum")
        plt.title(f"{self.d_list}")
        plt.plot(self.freqs, 20 * np.log10(np.abs(tmm_fd[:, 1])), label="r_tmm")
        plt.axvline(self.freqs[f0_i], color="red", label="Opt. range", linewidth=5.0)
        plt.axvline(self.freqs[f1_i], color="red", linewidth=5.0)

        plt.figure("Phase")
        plt.title(f"{self.d_list}")
        plt.plot(self.freqs, np.angle(tmm_fd[:, 1] / self.ref_fd[:, 1]), label="tmm")
        plt.axvline(self.freqs[f0_i], color="red", label="Opt. range", linewidth=5.0)
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


def main():
    from functools import partial

    df = 0.014275517487508922
    f0_idx = int(0.150 / df)
    f1_idx = int(4.000 / df)

    d0 = array([44.0, 650.0, 71.0])
    cost_inst = Cost(d_lst=d0, freq_idx_range=(f0_idx, f1_idx))

    # f_idx = int(1.68 / 0.014275517487508922)
    f_idx = 80
    print(f"Freq: {cost_inst.freqs[f_idx]} THz (idx: {f_idx})")

    #cost = partial(cost_inst.cost, freq_idx=f_idx)
    cost = cost_inst.cost_sm

    p0 = array([1.50120065, 2.86606279, 1.55, 0.01142857])
    p0 = array([0.4, 0.06, 0.4, 2.50e-3, 0.8e-3, 40,
                1.4, 0.55, 5.0, 5.5e-3, 1.3e-2, 310,
                0.4, 0.06, 0.4, 2.50e-3, 0.8e-3, 40])
    p0 = array([0.6, 0.75, 250, 203,
                3.8, 3.8, 5.50e-3, 7.8e-3,])

    res = cost(p=p0)
    cost_inst.plot_n(p0, sm=True)
    cost_inst.plot_model(p0, sm=True)

    print(res)


if __name__ == '__main__':
    main()
    plt.show()
