import numpy as np
from main import load_data
from functions import filtering, do_fft
from model.refractive_index import get_n
import matplotlib.pyplot as plt
from mpl_settings import *
from model.tmm_package import tmm_package_wrapper
from consts import *


class Cost:
    def __init__(self, sam_idx=None):
        self.ref_td, self.sam_td = load_data(sam_idx=sam_idx, signal_shift=-5)

        en_filter = False
        if en_filter:
            self.ref_td = filtering(self.ref_td, filt_type="hp", wn=2.3)
            self.sam_td = filtering(self.sam_td, filt_type="hp", wn=2.3)

        self.sam_fd, self.ref_fd = do_fft(self.sam_td), do_fft(self.ref_td)
        self.r_exp = self.sam_fd / self.ref_fd

        self.freqs = self.ref_fd[:, 0].real

        #self.plot_sam()

    def cost(self, p, freq_idx, en_plotting=True):
        print(self.freqs[freq_idx])
        d_list = [42.0, 641.0, 74.0]
        k = np.linspace(0, 0.1, len(self.freqs))

        n = array(p) + (np.array([0, 1, 0]) * 1j * k[freq_idx])

        r_tmm = tmm_package_wrapper(self.freqs[freq_idx], d_list, n)
        r_tmm[1] = r_tmm[1] * -1

        amp_loss = (np.abs(self.sam_fd[freq_idx, 1]) - np.abs(r_tmm[1] * self.ref_fd[freq_idx, 1])) ** 2
        phi_loss = (np.angle(r_tmm[1]) - np.angle(self.r_exp[freq_idx, 1])) ** 2

        loss = amp_loss + phi_loss

        if not en_plotting:
            return loss
        else:
            r_lst = []
            for i in range(len(self.freqs)):
                r_tmm = tmm_package_wrapper(self.freqs[i], d_list, n)
                r_tmm[1] = r_tmm[1] * -1
                r_lst.append(r_tmm[1])
            r_arr = np.array(r_lst)

            plt.figure("Spectrum")
            plt.plot(self.freqs, 20 * np.log10(np.abs(r_arr*self.ref_fd[:, 1])), label="r_tmm")
            plt.plot(self.freqs, 20 * np.log10(np.abs(self.ref_fd[:, 1])), label="ref")
            plt.plot(self.freqs, 20 * np.log10(np.abs(self.sam_fd[:, 1])), label="sam")
            plt.legend()

            plt.figure("Phase")
            plt.plot(self.freqs, np.angle(r_lst), label="tmm")
            plt.plot(self.freqs, np.angle(self.r_exp[:, 1]), label="r_exp")
            #plt.plot(self.freqs, np.angle(self.ref_fd[:, 1]), label="ref")
            #plt.plot(self.freqs, np.angle(self.sam_fd[:, 1]), label="sam")
            plt.legend()

            plt.show()



    def plot_sam(self):
        plt.figure("Time domain")
        plt.plot(self.ref_td[:, 0], self.ref_td[:, 1], label=f"Reference")
        plt.plot(self.sam_td[:, 0], self.sam_td[:, 1], label=f"Sample")
        # plt.plot(tmm_td[:, 0], tmm_td[:, 1], label=f"TMM * Reference")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

        plt.figure("Spectrum")
        plt.plot(self.ref_fd[:, 0], 20 * np.log10(np.abs(self.ref_fd[:, 1])), label=f"Reference")
        plt.plot(self.sam_fd[:, 0], 20 * np.log10(np.abs(self.sam_fd[:, 1])), label=f"Sample")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")
        plt.legend()

        plt.figure("Phase")
        # plt.plot(self.sam_fd[:, 0], np.angle(self.sam_fd[:, 1]), label="$\phi_{sam}$")
        # plt.plot(self.sam_fd[:, 0], np.angle(self.ref_fd[:, 1]), label="$\phi_{ref}$")
        plt.plot(self.sam_fd[:, 0], np.angle(self.r_exp), label="$\phi_{sam} - \phi_{ref}$")
        # plt.plot(self.sam_fd[:, 0], phase_tmm, label="$\phi_{TMM}$")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (Rad)")
        plt.legend()

        plt.legend(loc='upper right')


def main():
    cost = Cost().cost
    print(cost(p=[1.5, 2.8, 1.5], freq_idx=100))


if __name__ == '__main__':
    main()
