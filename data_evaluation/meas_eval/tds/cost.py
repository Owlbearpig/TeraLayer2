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
        self.p0 = p0

        if d_lst is None:
            self.d_list = [42.0, 641.0, 74.0]
        else:
            self.d_list = d_lst

        self.ref_td, self.sam_td = load_data(sam_idx=self.sam_idx, signal_shift=-5)

        en_filter = False
        if en_filter:
            self.ref_td = filtering(self.ref_td, filt_type="hp", wn=2.3)
            self.sam_td = filtering(self.sam_td, filt_type="hp", wn=2.3)

        self.sam_fd, self.ref_fd = do_fft(self.sam_td), do_fft(self.ref_td)

        self.freqs = self.ref_fd[:, 0].real
        self.t = self.ref_td[:, 0].real
        self.r_exp = array([self.freqs, self.sam_fd[:, 1] / self.ref_fd[:, 1]]).T

        self.k = np.array([np.linspace(0, 0.00, len(self.freqs)),
                           np.linspace(0, 0.10, len(self.freqs)),
                           np.linspace(0, 0.00, len(self.freqs))]).T

        if freq_idx_range is None:
            self.freq_idx_range = (0, len(self.freqs))
        else:
            self.freq_idx_range = freq_idx_range
        self.freq_range = self.freqs[freq_idx_range[0]:freq_idx_range[1]]

    def cost(self, p, freq_idx, return_both=False):
        n = array(p) + 1j * self.k[freq_idx]

        r_tmm = tmm_package_wrapper(self.freqs[freq_idx], self.d_list, n)
        r_tmm[1] = r_tmm[1] * -1
        r_exp = self.sam_fd[freq_idx, 1] / self.ref_fd[freq_idx, 1]

        amp_loss = (np.abs(r_exp) - np.abs(r_tmm[1])) ** 2
        phi_loss = (np.angle(r_tmm[1]) - np.angle(self.r_exp[freq_idx, 1])) ** 2

        loss = amp_loss + phi_loss

        if return_both:
            return amp_loss, phi_loss
        else:
            return loss

    def pad_n(self, p, freq_idx_range):
        if self.p0 is not None:
            pad = self.p0
        else:
            pad = [1.5, 2.8, 1.5]

        f0_i, f1_i = freq_idx_range
        pre_pad, post_pad = np.vstack(f0_i * [pad]), np.vstack((len(self.freqs) - f1_i) * [pad])
        n = np.concatenate((pre_pad, p, post_pad)) + 1j * self.k

        return n

    def calc_model(self, p):
        n = self.pad_n(p, self.freq_idx_range)
        r_tmm_fd = tmm_package_wrapper(self.freqs, self.d_list, n)
        r_tmm_fd[:, 1] *= -1

        tmm_fd = array([self.freqs, r_tmm_fd[:, 1] * self.ref_fd[:, 1]]).T
        tmm_td = do_ifft(tmm_fd)

        return tmm_fd, tmm_td

    def gof(self, p):
        res = {}
        tmm_fd, tmm_td = self.calc_model(p)
        res["peas_corr_coeff"] = pearsonr(self.sam_td[:, 1].real, tmm_td[:, 1].real)

        return res

    def plot_padded_n(self, p):
        f0_i, f1_i = self.freq_idx_range
        n = self.pad_n(p, self.freq_idx_range)
        plt.figure("Refractive index real")
        plt.title(f"{self.d_list}")
        plt.plot(self.freqs, n[:, 0].real, label="Re($n_0$)")
        plt.plot(self.freqs, n[:, 1].real, label="Re($n_1$)")
        plt.plot(self.freqs, n[:, 2].real, label="Re($n_2$)")
        plt.axvline(self.freqs[f0_i], color="red", label="Opt. range", linewidth=5.0)
        plt.axvline(self.freqs[f1_i], color="red", linewidth=5.0)
        plt.legend()

        plt.figure("Refractive index imag")
        plt.title(f"{self.d_list}")
        plt.plot(self.freqs, n[:, 0].imag, label="Img($n_0$)")
        plt.plot(self.freqs, n[:, 1].imag, label="Img($n_1$)")
        plt.plot(self.freqs, n[:, 2].imag, label="Img($n_2$)")
        plt.axvline(self.freqs[f0_i], color="red", label="Opt. range", linewidth=5.0)
        plt.axvline(self.freqs[f1_i], color="red", linewidth=5.0)
        plt.legend()

    def plot_model(self, p):
        f0_i, f1_i = self.freq_idx_range
        tmm_fd, tmm_td = self.calc_model(p)

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

    cost_inst = Cost()
    f_idx = int(1.68 / 0.014275517487508922)
    print(f"frequency: {cost_inst.freqs[f_idx]} THz (idx: {f_idx})")
    cost = partial(cost_inst.cost, freq_idx=f_idx)

    p0 = [1.5, 2.8, 1.5]
    res = cost(p=p0)

    print(res)


if __name__ == '__main__':
    main()
