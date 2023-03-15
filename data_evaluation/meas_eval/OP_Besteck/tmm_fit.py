import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from model.tmm_package import tmm_package_wrapper
from functions import do_ifft, do_fft
from load_data import OPMeasurement
from scipy.optimize import shgo


class SamplePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coords = np.array([self.x, self.y])

    def __getitem__(self, item):
        return self.coords[item]

    def __repr__(self):
        return f"x={self[0]} mm, y={self[1]} mm"


def shift_ref(ref_td, sam_td):
    pos_pulse1_sam = np.argmax(sam_td[int(17 / 0.05):int(20 / 0.05), 1])
    pos_pulse1_ref = np.argmax(ref_td[int(17 / 0.05):int(20 / 0.05), 1])

    idx_shift = pos_pulse1_sam - pos_pulse1_ref

    shifted_ref_td = array([ref_td[:, 0], np.roll(ref_td[:, 1], idx_shift)]).T

    return shifted_ref_td


def analytical_eval(r_exp):
    # not finished / incorrect. Possibly refer to:
    # THz-TDS Reflection Measurement of Coating Thicknesses at
    # Non-Perpendicular Incidence: Experiment and Simulation
    freqs = r_exp[:, 0]

    angle = 8 * np.pi / 180
    n_enum = 1 - np.abs(r_exp[:, 1]) ** 2
    n_denum = 1 + np.abs(r_exp[:, 1]) ** 2 - 2 * np.abs(r_exp[:, 1]) * np.cos(angle)

    n = n_enum / n_denum

    plt.figure("Refractive index")
    plt.plot(freqs, n, label="analytical Real part")
    plt.legend()

    return n


def fitting(ref_fd, sam_fd):
    freqs = sam_fd[:, 0].real

    d = array([260, 1e10])

    def cost(n, freq_idx, absorp=False):
        n = np.array([n, 500])

        r_tmm = tmm_package_wrapper(freqs[freq_idx], d, n)

        phi_exp = np.angle(sam_fd[freq_idx, 1])

        phi_mod = np.angle(ref_fd[freq_idx, 1] * r_tmm[1])

        loss = (phi_exp - phi_mod) ** 2

        amp_loss = (np.abs(sam_fd[freq_idx, 1]) - np.abs(ref_fd[freq_idx, 1] * r_tmm[1])) ** 2

        if absorp:
            loss += amp_loss

        return loss

    def cost_absorption(k, freq_idx, n, absorp=True):
        n = np.array([n + 1j * k, 500])

        r_tmm = tmm_package_wrapper(freqs[freq_idx], d, n)

        phi_exp = np.angle(sam_fd[freq_idx, 1])

        phi_mod = np.angle(ref_fd[freq_idx, 1] * r_tmm[1])

        loss = (phi_exp - phi_mod) ** 2

        amp_loss = (np.abs(sam_fd[freq_idx, 1]) - np.abs(ref_fd[freq_idx, 1] * r_tmm[1])) ** 2

        if absorp:
            loss += amp_loss

        return loss

    # """
    # simple fit, only phase no absorption
    bounds = array([1.8, 2.1])
    ret = []
    for freq_idx, freq in enumerate(freqs):
        print(freq, freq_idx)
        min_val, min_n = np.inf, None
        for n in np.arange(bounds[0], bounds[1], 0.01):
            loss = cost(n, freq_idx)
            if loss < min_val:
                min_val = loss
                min_n = n
        print(freq, freq_idx, min_n)
        ret.append(min_n)
    # """
    ret = array(ret)

    bounds = array([0.001, 0.100])
    ret_k = []
    for freq_idx, freq in enumerate(freqs):
        min_val, min_k = np.inf, None
        for k in np.arange(bounds[0], bounds[1], 0.001):
            loss = cost_absorption(k, freq_idx, ret[freq_idx])
            if loss < min_val:
                min_val = loss
                min_k = k
        print(freq, freq_idx, min_k)
        ret_k.append(min_k)

    ret_k = array(ret_k)

    """
    # 2D fit (considering absorption)
    bounds = array([(1.8, 2.1), (0.001, 0.100)])
    ret = []
    for freq_idx, freq in enumerate(freqs):
        print(freq, freq_idx)
        res = shgo(cost, bounds, args=(freq_idx, True), iters=4)
        ret.append(res.x[0] + 1j*res.x[1])

    ret = array(ret)
    """

    plt.figure("Refractive index")
    plt.plot(freqs, ret, label="Real part")
    plt.plot(freqs, ret_k, label="Imag part")
    plt.legend()

    return ret + 1j * ret_k


def main():
    measurement = OPMeasurement(area_idx=1)
    point = SamplePoint(x=7.0, y=4.0)
    point_metal = SamplePoint(x=1.0, y=4.0)

    # ref_td, ref_fd = measurement.get_ref(both=True)
    ref_td = measurement.get_point(x=point_metal[0], y=point_metal[1])
    sam_td, sam_fd = measurement.get_point(x=point[0], y=point[1], both=True)

    ref_td = shift_ref(ref_td, sam_td)
    ref_fd = do_fft(ref_td)

    freqs = ref_fd[:, 0].real
    ones = np.ones_like(freqs)

    r_exp_fd = array([ref_fd[:, 0], sam_fd[:, 1] / ref_fd[:, 1]]).T

    n = fitting(ref_fd, sam_fd)

    # n = np.array([1.99 * ones, 500 * ones]).T
    n = np.array([n, 500 * ones]).T
    d = array([260, 1e10])

    r_tmm_fd = tmm_package_wrapper(freqs, d, n)
    sam_tmm_fd = array([freqs, ref_fd[:, 1] * r_tmm_fd[:, 1]]).T

    plt.figure("Spectrum")
    plt.plot(freqs, 20 * np.log10(np.abs(ref_fd[:, 1])), label=f"Reference {point_metal} (shifted)")
    plt.plot(freqs, 20 * np.log10(np.abs(sam_fd[:, 1])), label=f"Sample {point}")
    plt.plot(freqs, 20 * np.log10(np.abs(ref_fd[:, 1] * r_tmm_fd[:, 1])), label=f"Ref * r_TMM")
    plt.plot(freqs, 20 * np.log10(np.abs(r_exp_fd[:, 1])), label="r_exp")
    plt.plot(freqs, 20 * np.log10(np.abs(r_tmm_fd[:, 1])), label="r_TMM")
    plt.legend()
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude (dB)")

    sam_tmm_td = do_ifft(sam_tmm_fd)

    plt.figure("Time domain")
    plt.plot(ref_td[:, 0], ref_td[:, 1], label=f"Reference {point_metal} (shifted)")
    plt.plot(sam_td[:, 0], sam_td[:, 1], label=f"Sample ({point})")
    plt.plot(sam_tmm_td[:, 0], sam_tmm_td[:, 1], label="Sample model (Ref * r_TMM)")
    plt.legend()
    plt.xlabel("Time (ps)")
    plt.ylabel("Amplitude (Arb. u.)")

    plt.figure("Phase")
    # plt.plot(freqs, np.angle(ref_fd[:, 1]), label=f"Reference {point_metal} (shifted)")
    plt.plot(freqs, np.angle(sam_fd[:, 1]), label=f"Sample {point}")
    plt.plot(freqs, np.angle(ref_fd[:, 1] * r_tmm_fd[:, 1]), label=f"Ref * r_TMM")
    # plt.plot(freqs, np.angle(r_tmm[:, 1]), label="r_TMM")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (Rad)")
    plt.legend()


if __name__ == '__main__':
    main()
    plt.show()
