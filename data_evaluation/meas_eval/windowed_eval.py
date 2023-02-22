import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import shgo, curve_fit
from consts import *
from model.tmm_package import tmm_package_wrapper
from functions import do_ifft
from functools import partial
from measurement import Measurement
from tmm import coh_tmm


def simple_fit(freq_idx_range, measurement):
    sam_idx = measurement.sam_idx
    ref_td, sam_td, ref_fd, sam_fd = measurement.load_measurement(tds_data=True, en_window=True)
    r_exp = sam_fd[:, 1] / ref_fd[:, 1]

    # d_list = array([np.inf, 41.0, np.inf])
    d_list = array([np.inf, 41.0, np.inf])
    # d_list = array([np.inf, 65.0, np.inf])
    angle_in = 8 * pi / 180

    m = freq_idx_range[1] - freq_idx_range[0]  # freq_cnt
    f_opt_amp, f_opt_phi, n_res = np.zeros(m), np.zeros(m), np.zeros(m)
    for loop_idx, freq_idx in enumerate(range(*freq_idx_range)):
        best_n, min_val = None, np.inf
        all_losses = np.zeros(1000)
        n_arr = np.linspace(1.2, 1.8, 1000)
        for i, n1 in enumerate(n_arr):
            freq = ref_fd[freq_idx, 0].real

            lambda_vac = (c0 / freq) * 10 ** -6
            n = [1.0, n1, 2.9]
            # n = [2.9, n1, 1]
            r_tmm = coh_tmm("s", n, d_list, angle_in, lambda_vac)["r"] * -1

            amp_loss = (np.log10(np.abs(r_tmm)) - np.log10(np.abs(r_exp[freq_idx]))) ** 2
            phi_loss = 0 * (np.angle(r_tmm) - np.angle(r_exp[freq_idx])) ** 2

            loss = amp_loss + phi_loss
            all_losses[i] = loss
            if loss < min_val:
                min_val = loss
                best_n = n1

        if np.isclose(0.4282655246252677, ref_fd[freq_idx, 0].real):
            plt.figure("loss")
            plt.plot(n_arr, all_losses, label="single layer model")

        n_res[loop_idx] = best_n
        print(f"Freq: {freq} (Idx: {freq_idx}), res.x: {best_n}, res.fun: {loss}")

    f0_idx, f1_idx = freq_idx_range

    r_tmm_fd = []
    for freq_idx in range(len(ref_fd[:, 0].real)):
        lambda_vac = (c0 / ref_fd[freq_idx, 0].real) * 10 ** -6
        if freq_idx < f0_idx:
            n = [1.0, n_res[0], 2.9]
        elif (freq_idx >= f0_idx) and (freq_idx < f1_idx):
            n = [1.0, n_res[freq_idx - f0_idx], 2.9]
        else:
            n = [1.0, n_res[-1], 2.9]

        r_tmm = coh_tmm("s", n, d_list, angle_in, lambda_vac)["r"] * -1
        r_tmm_fd.append(r_tmm)
    """
    r_tmm_fd = []
    for loop_idx, freq_idx in enumerate(range(*freq_idx_range)):
        lambda_vac = (c0 / ref_fd[freq_idx, 0].real) * 10 ** -6
        n = [1.0, n_res[loop_idx], 2.9]
        r_tmm = coh_tmm("s", n, d_list, angle_in, lambda_vac)["r"] * -1
        r_tmm_fd.append(r_tmm)
    """

    en_plotts_ = False
    if en_plotts_:

        t = ref_td[:, 0].real
        freqs = ref_fd[:, 0].real
        f_idx0, f_idx1 = freq_idx_range

        r_tmm_fd = np.array(r_tmm_fd)

        r_sam = r_tmm_fd * ref_fd[:, 1]
        r_sam = array([freqs, r_sam]).T
        tmm_td = do_ifft(r_sam, shift=0)

        r_tmm_fd = array([freqs, r_tmm_fd]).T

        if sam_idx is None:
            name = "Avg."
        else:
            name = sam_idx

        plt.figure("Time domain")
        plt.title(f"Layers: {d_list} $(\mu m)$")
        plt.plot(t, ref_td[:, 1], label=f"Ref. (Meas. idx: {name})")
        plt.plot(t, sam_td[:, 1], label=f"Sam. (Meas. idx: {name})")
        plt.plot(tmm_td[:, 0].real, tmm_td[:, 1], label=f"TMM")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

        plt.figure("Spectrum")
        plt.title(f"Layers: {d_list} $(\mu m)$")
        # plt.plot(self.freqs, 20 * np.log10(np.abs(self.ref_fd[:, 1])), label=f"Ref. (Meas. idx: {self.sam_idx})")
        # plt.plot(self.freqs, 20 * np.log10(np.abs(self.sam_fd[:, 1])), label=f"Sam. (Meas. idx: {self.sam_idx})")
        plt.plot(freqs, 20 * np.log10(np.abs(ref_fd[:, 1])), label=f"Ref. (Meas. idx: {name})")
        plt.plot(freqs, 20 * np.log10(np.abs(sam_fd[:, 1])), label=f"Sam. (Meas. idx: {name})")
        plt.plot(freqs[f_idx0:f_idx1], 20 * np.log10(np.abs(r_tmm_fd[f_idx0:f_idx1, 1] * ref_fd[f_idx0:f_idx1, 1])), label="TMM")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")
        plt.legend()

        plt.figure("Phase")
        plt.title(f"Layers: {d_list} $(\mu m)$")
        # plt.plot(self.sam_fd[:, 0], np.angle(self.sam_fd[:, 1]), label="$\phi_{sam}$")
        # plt.plot(self.sam_fd[:, 0], np.angle(self.ref_fd[:, 1]), label="$\phi_{ref}$")
        plt.plot(sam_fd[:, 0], np.angle(r_exp), label=f"(Sam idx: {name}) " + "$\phi_{sam} - \phi_{ref}$")
        plt.plot(freqs[f_idx0:f_idx1], np.angle(r_tmm_fd[f_idx0:f_idx1, 1]), label="TMM")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (Rad)")
        plt.legend(loc='upper right')

        plt.figure("Refractive index")
        plt.title(f"Layers: {d_list} $(\mu m)$")
        plt.plot(freqs[f_idx0:f_idx1], n_res)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Refractive index (Real)")

    return n_res


def full_fit(n_outer, freq_idx_range, measurement):
    sam_idx = measurement.sam_idx
    ref_td, sam_td, ref_fd, sam_fd = measurement.load_measurement(tds_data=True, en_window=False)
    r_exp = sam_fd[:, 1] / ref_fd[:, 1]

    angle_in = 8 * pi / 180
    # d_list = array([np.inf, 41.0, 642.0, 64.0, np.inf])
    # d_list = array([np.inf, 41.0, 641.0, 64.0, np.inf]) # best
    d_list_ = array([np.inf, 41.0, 646.0, 62.0, np.inf])

    std0_min, std1_min = None, None
    std0_min_val, std1_min_val = np.inf, np.inf
    for d1 in range(-10, 11):
        for d2 in range(-10, 11):
            if (d1, d2) != (-2, 5):
                continue

            print("Progress: ", d1, d2)
            d_list = d_list_.copy() + array([0, 0, d1, d2, 0])

            m = freq_idx_range[1] - freq_idx_range[0]  # freq_cnt
            f_opt_amp, f_opt_phi, n_res = np.zeros(m), np.zeros(m), np.zeros((m, 2))

            def cost(p, freq):

                lambda_vac = (c0 / freq) * 10 ** -6
                n = [1.0, n_outer[loop_idx], p[0], p[1], 1]
                r_tmm = coh_tmm("s", n, d_list, angle_in, lambda_vac)["r"] * -1

                amp_loss = ((np.abs(r_tmm)) - (np.abs(r_exp[freq_idx]))) ** 2
                phi_loss = (np.angle(r_tmm) - np.angle(r_exp[freq_idx])) ** 2

                loss = amp_loss + phi_loss

                return loss

            for loop_idx, freq_idx in enumerate(range(*freq_idx_range)):
                bounds = array([(2.7, 2.95), (1.2, 1.8)])
                freq = ref_fd[freq_idx, 0].real
                res = shgo(func=cost, bounds=bounds, iters=5, args=(freq,))
                n_res[loop_idx, ] = res.x
                print(f"Freq: {freq} (Idx: {freq_idx}), res.x: {res.x}, res.fun: {res.fun}")

                """
                best_n, min_val = None, np.inf
                all_losses = np.zeros(1000)
                phi_losses, amp_losses = np.zeros(1000), np.zeros(1000)
                
                n_arr = np.linspace(2.7, 2.95, 1000)
                for i, n1 in enumerate(n_arr):
                    freq = ref_fd[freq_idx, 0].real
        
                    lambda_vac = (c0 / freq) * 10 ** -6
                    n = [1.0, n_outer[loop_idx], n1, n_outer[loop_idx], 1]
                    r_tmm = coh_tmm("s", n, d_list, angle_in, lambda_vac)["r"] * -1
        
                    amp_loss = ((np.abs(r_tmm)) - (np.abs(r_exp[freq_idx]))) ** 2
                    phi_loss = (np.angle(r_tmm) - np.angle(r_exp[freq_idx])) ** 2
        
                    loss = amp_loss + phi_loss
                    all_losses[i] = loss
                    phi_losses[i] = phi_loss
                    amp_losses[i] = amp_loss
                    if loss < min_val:
                        min_val = loss
                        best_n = n1
        
                if freq_idx in [37, 54, 51]:
                    plt.figure("loss")
                    plt.title(f"{d_list}")
                    plt.plot(n_arr, all_losses, label=f"sum {ref_fd[freq_idx, 0].real}")
                    plt.plot(n_arr, phi_losses, label=f"phi loss {ref_fd[freq_idx, 0].real}")
                    plt.plot(n_arr, amp_losses, label=f"amp loss {ref_fd[freq_idx, 0].real}")
                    plt.legend()
        
                n_res[loop_idx] = best_n
                print(f"Freq: {freq} (Idx: {freq_idx}), res.x: {best_n}, res.fun: {loss}")
                """
            #std0, std1 = sum(np.abs(np.diff(n_res[:, 0]))), sum(np.abs(np.diff(n_res[:, 1])))
            std0, std1 = np.std(n_res[:, 0]), np.std(n_res[:, 1])

            print(f"STDs p=(n0, n1): {std0}, {std1}")
            if std0 < std0_min_val:
                std0_min = [d1, d2]
                std0_min_val = std0
                print(f"std0 best fit: {std0_min}, ({std0_min_val})")

            if std1 < std1_min_val:
                std1_min = [d1, d2]
                std1_min_val = std1
                print(f"std1 best fit: {std1_min}, ({std1_min_val})")

    print(f"std0 best fit: {std0_min}, ({std0_min_val})")
    print(f"std1 best fit: {std1_min}, ({std1_min_val})")

    en_plotts_ = True
    if en_plotts_:
        t = ref_td[:, 0].real
        freqs = ref_fd[:, 0].real
        f_idx0, f_idx1 = freq_idx_range

        r_tmm_fd = []
        for freq_idx in range(len(ref_fd[:, 0].real)):
            lambda_vac = (c0 / ref_fd[freq_idx, 0].real) * 10 ** -6
            if freq_idx < f_idx0:
                n = [1.0, n_outer[0], n_res[0, 0], n_res[0, 1], 1.0]
            elif (freq_idx >= f_idx0) and (freq_idx < f_idx1):
                n = [1.0, n_outer[freq_idx - f_idx0], n_res[freq_idx - f_idx0, 0], n_res[freq_idx - f_idx0, 1], 1.0]
            else:
                n = [1.0, n_outer[-1], n_res[-1, 0], n_res[-1, 1], 1.0]

            r_tmm = coh_tmm("s", n, d_list, angle_in, lambda_vac)["r"] * -1
            r_tmm_fd.append(r_tmm)

        """
        r_tmm_fd = []
        for loop_idx, freq_idx in enumerate(range(*freq_idx_range)):
            lambda_vac = (c0 / ref_fd[freq_idx, 0].real) * 10 ** -6
            n = [1.0, n_outer[loop_idx], n_res[loop_idx, 0], n_res[loop_idx, 1], 1.0]
            r_tmm = coh_tmm("s", n, d_list, angle_in, lambda_vac)["r"] * -1
            r_tmm_fd.append(r_tmm)
        """

        r_tmm_fd = np.array(r_tmm_fd)

        r_sam = r_tmm_fd * ref_fd[:, 1]
        r_sam = array([freqs, r_sam]).T
        tmm_td = do_ifft(r_sam, shift=0)

        r_tmm_fd = array([freqs, r_tmm_fd]).T

        plt.figure("Time domain")
        plt.title(f"Layers: {d_list} $(\mu m)$")
        plt.plot(t, ref_td[:, 1], label=f"Ref. (Meas. idx: {sam_idx})")
        plt.plot(t, sam_td[:, 1], label=f"Sam. (Meas. idx: {sam_idx})")
        plt.plot(tmm_td[:, 0].real, tmm_td[:, 1], label=f"TMM")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

        plt.figure("Spectrum")
        plt.title(f"Layers: {d_list} $(\mu m)$")
        # plt.plot(self.freqs, 20 * np.log10(np.abs(self.ref_fd[:, 1])), label=f"Ref. (Meas. idx: {self.sam_idx})")
        # plt.plot(self.freqs, 20 * np.log10(np.abs(self.sam_fd[:, 1])), label=f"Sam. (Meas. idx: {self.sam_idx})")
        plt.plot(freqs, 20 * np.log10(np.abs(ref_fd[:, 1])), label=f"Ref. (Meas. idx: {sam_idx})")
        plt.plot(freqs, 20 * np.log10(np.abs(sam_fd[:, 1])), label=f"Sam. (Meas. idx: {sam_idx})")
        plt.plot(freqs[f_idx0:f_idx1], 20 * np.log10(np.abs(r_tmm_fd[f_idx0:f_idx1, 1] * ref_fd[f_idx0:f_idx1, 1])), label="TMM")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")
        plt.legend()

        plt.figure("Phase")
        plt.title(f"Layers: {d_list} $(\mu m)$")
        # plt.plot(self.sam_fd[:, 0], np.angle(self.sam_fd[:, 1]), label="$\phi_{sam}$")
        # plt.plot(self.sam_fd[:, 0], np.angle(self.ref_fd[:, 1]), label="$\phi_{ref}$")
        plt.plot(sam_fd[:, 0], np.angle(r_exp), label=f"(Sam idx: {sam_idx}) " + "$\phi_{sam} - \phi_{ref}$")
        plt.plot(freqs[f_idx0:f_idx1], np.angle(r_tmm_fd[f_idx0:f_idx1, 1]), label="TMM")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (Rad)")
        plt.legend(loc='upper right')

        plt.figure("Refractive index")
        plt.title(f"Layers: {d_list} $(\mu m)$")
        plt.plot(freqs[f_idx0:f_idx1], n_outer, label="RI first layer")
        plt.plot(freqs[f_idx0:f_idx1], n_res[:, 0], label="RI second layer")
        plt.plot(freqs[f_idx0:f_idx1], n_res[:, 1], label="RI third layer")
        plt.legend()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Refractive index (Real)")

    return n_res, d_list


def main():
    df = 0.014275517487508922  # tds
    # f0_idx = int(0.350 / df)
    # f1_idx = int(2.500 / df)
    f0_idx = int(0.400 / df)
    f1_idx = int(1.050 / df)
    #f0_idx = int(0.400 / df)
    #f1_idx = int(0.550 / df)

    freq_idx_range = f0_idx, f1_idx

    np.random.seed(420)
    sam_idx = np.random.randint(0, 100)
    #sam_idx = None
    print(f"sam_idx: {sam_idx}")
    measurement = Measurement(sam_idx)

    n_res = simple_fit(freq_idx_range, measurement)
    n_full, d_list = full_fit(n_res, freq_idx_range, measurement)

    """
    n_avg = np.zeros((f1_idx - f0_idx, 2))
    for sam_idx in range(0, 99):

        print(f"sam_idx: {sam_idx}")
        measurement = Measurement(sam_idx)

        n_res = simple_fit(freq_idx_range, measurement)
        n_full, d_list = full_fit(n_res, freq_idx_range, measurement)

        n_avg += n_full

    n_avg /= 99
    
    freqs = measurement.ref_fd[:, 0].real

    plt.figure("Refractive index")
    plt.title(f"{d_list}")
    plt.plot(freqs[f0_idx:f1_idx], n_avg[:, 0])
    plt.plot(freqs[f0_idx:f1_idx], n_avg[:, 1])
    """


if __name__ == '__main__':
    main()
    plt.show()
