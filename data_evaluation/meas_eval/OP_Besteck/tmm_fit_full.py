import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from model.tmm_package import tmm_package_wrapper
from functions import do_ifft, do_fft
from load_data import OPMeasurement
from scipy.optimize import shgo


def fitting(ref_fd, sam_fd):
    freqs = sam_fd[:, 0].real


    d = array([140, 1e10])

    point = measurement.position
    d_film = sample_thicknesses[self.sample_idx]
    d_list = [inf, d_sub, d_film, inf]

    film_td = measurement.get_data_td()
    film_ref_td = self.get_ref(both=False, coords=point)

    film_td = window(film_td, win_len=12, shift=0, en_plot=False, slope=0.05)
    film_ref_td = window(film_ref_td, win_len=12, shift=0, en_plot=False, slope=0.05)

    film_ref_fd, film_fd = do_fft(film_ref_td), do_fft(film_td)

    film_ref_fd = phase_correction(film_ref_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)
    film_fd = phase_correction(film_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)

    freqs = film_ref_fd[:, 0].real
    omega = 2 * pi * freqs
    f_idx = np.argmin(np.abs(freqs - selected_freq_))

    def cost_both(p, f_idx):

        n = array([1, p[0] + 1j * p[1], 500.0+1j*500.0, 1])
        lam_vac = c_thz / freqs[f_idx]
        t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)["t"]
        sam_tmm_fd = t_tmm_fd * film_ref_fd[f_idx, 1] * phase_shift[f_idx]

        amp_loss = (np.abs(sam_tmm_fd) - np.abs(film_fd[f_idx, 1])) ** 2
        phi_loss = (np.angle(sam_tmm_fd) - np.angle(film_fd[f_idx, 1])) ** 2

        return amp_loss + phi_loss

    bounds = shgo_bounds_film[self.sample_idx]
    iters = 7
    res = shgo(cost_both, bounds=bounds, iters=iters - 2)
    while res.fun > 1e-5:
        iters += 1
        res = shgo(cost, bounds=bounds, iters=iters)
        if iters >= 7:
            break

    n_opt = res.x[0] + 1j * res.x[1]

    plt.figure("Refractive index")
    plt.plot(freqs, n_opt.real, label="Real part")
    plt.plot(freqs, n_opt.imag, label="Imag part")
    plt.legend()


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
    d = array([140, 1e10])

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
