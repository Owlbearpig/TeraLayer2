import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import shgo, curve_fit
from consts import *
from model.tmm_package import tmm_package_wrapper
from functions import do_ifft
from measurement import Measurement
from tmm import coh_tmm


def simple_fit(freq_idx_range):
    np.random.seed(420)
    sam_idx = np.random.randint(0, 100)
    print(f"sam_idx: {sam_idx}")
    measurement = Measurement(sam_idx)
    ref_td, sam_td, ref_fd, sam_fd = measurement.load_measurement(tds_data=True, en_window=True)
    r_exp = sam_fd[:, 1] / ref_fd[:, 1]

    #d_list = array([np.inf, 41.0, np.inf])
    d_list = array([np.inf, 41.0, np.inf])
    #d_list = array([np.inf, 71.0, np.inf])
    angle_in = 8 * pi / 180

    m = freq_idx_range[1] - freq_idx_range[0]  # freq_cnt
    f_opt_amp, f_opt_phi, n_res = np.zeros(m), np.zeros(m), np.zeros(m)
    for loop_idx, freq_idx in enumerate(range(*freq_idx_range)):
        best_n, min_val = None, np.inf
        all_losses = np.zeros(1000)
        for i, n1 in enumerate(np.linspace(1.2, 1.8, 1000)):
            freq = ref_fd[freq_idx, 0].real

            lambda_vac = (c0 / freq) * 10 ** -6
            n = [1.0, n1, 2.9]
            r_tmm = coh_tmm("s", n, d_list, angle_in, lambda_vac)["r"] * -1

            amp_loss = (np.log10(np.abs(r_tmm)) - np.log10(np.abs(r_exp[freq_idx]))) ** 2
            phi_loss = (np.angle(r_tmm) - np.angle(r_exp[freq_idx])) ** 2

            loss = amp_loss + phi_loss
            all_losses[i] = loss
            if loss < min_val:
                min_val = loss
                best_n = n1

        if np.isclose(0.4282655246252677, ref_fd[freq_idx, 0].real):
            plt.figure("loss")
            plt.plot(all_losses)

        n_res[loop_idx] = best_n
        print(f"Freq: {freq} (Idx: {freq_idx}), res.x: {best_n}, res.fun: {loss}")

    r_tmm_fd = []
    for loop_idx, freq_idx in enumerate(range(*freq_idx_range)):
        lambda_vac = (c0 / ref_fd[freq_idx, 0].real) * 10 ** -6
        n = [1.0, n_res[loop_idx], 2.9]
        r_tmm = coh_tmm("s", n, d_list, angle_in, lambda_vac)["r"] * -1
        r_tmm_fd.append(r_tmm)

    t = ref_td[:, 0].real
    freqs = ref_fd[:, 0].real
    f_idx0, f_idx1 = freq_idx_range

    r_tmm_fd = np.array(r_tmm_fd)
    r_tmm_fd = array([freqs[f_idx0:f_idx1], r_tmm_fd]).T
    tmm_td = do_ifft(r_tmm_fd, shift=0)

    plt.figure("Time domain")
    plt.plot(t, ref_td[:, 1], label=f"Ref. (Meas. idx: {sam_idx})")
    plt.plot(t, sam_td[:, 1], label=f"Sam. (Meas. idx: {sam_idx})")
    plt.plot(tmm_td[:, 0].real, tmm_td[:, 1], label=f"TMM")
    plt.xlabel("Time (ps)")
    plt.ylabel("Amplitude (nA)")
    plt.legend()

    plt.figure("Spectrum")
    # plt.plot(self.freqs, 20 * np.log10(np.abs(self.ref_fd[:, 1])), label=f"Ref. (Meas. idx: {self.sam_idx})")
    # plt.plot(self.freqs, 20 * np.log10(np.abs(self.sam_fd[:, 1])), label=f"Sam. (Meas. idx: {self.sam_idx})")
    plt.plot(freqs[f_idx0:f_idx1], 20 * np.log10(np.abs(r_tmm_fd[:, 1] * ref_fd[f_idx0:f_idx1, 1])), label="r_tmm")
    plt.plot(freqs, 20 * np.log10(np.abs(ref_fd[:, 1])), label=f"Ref. (Meas. idx: {sam_idx})")
    plt.plot(freqs, 20 * np.log10(np.abs(sam_fd[:, 1])), label=f"Sam. (Meas. idx: {sam_idx})")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude (dB)")
    plt.legend()

    plt.figure("Phase")
    # plt.plot(self.sam_fd[:, 0], np.angle(self.sam_fd[:, 1]), label="$\phi_{sam}$")
    # plt.plot(self.sam_fd[:, 0], np.angle(self.ref_fd[:, 1]), label="$\phi_{ref}$")
    plt.plot(freqs[f_idx0:f_idx1], np.angle(r_tmm_fd[:, 1]), label="r_tmm")
    plt.plot(sam_fd[:, 0], np.angle(r_exp), label=f"(Sam idx: {sam_idx}) " + "$\phi_{sam} - \phi_{ref}$")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (Rad)")
    plt.legend(loc='upper right')

    plt.figure("Refractive index")
    plt.plot(freqs[f_idx0:f_idx1], n_res)

    return n_res


def main():
    df = 0.014275517487508922  # tds
    f0_idx = int(0.350 / df)
    f1_idx = int(2.500 / df)

    freq_idx_range = f0_idx, f1_idx

    n_res = simple_fit(freq_idx_range)


if __name__ == '__main__':
    main()
    plt.show()
