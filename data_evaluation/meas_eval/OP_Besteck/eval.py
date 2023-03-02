import numpy as np
from load_data import OPMeasurement
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, norm
from scipy.fftpack import rfft
from consts import c_thz
from mpl_settings import *


def deconvolve(ref_td, sam_td):
    def shrinkage_factor(H):
        H = np.array(H, dtype=np.float32)
        H = np.dot(H, H.T)
        return 1 / max(rfft(H[0]))  # since H is circulant -> H.T @ H is -> eig vals with fft

    def toeplitz_j(ref_td):  # convolve(ref, y, mode="wrap") from scipy.ndimage.filters
        c = np.hstack((ref_td[:, 1], ref_td[0, 1]))
        c = toeplitz(c, c[-1::-1])
        return c[0:-1, 0:-1]

    lambda_ = 2
    eps = 1e-14
    max_iteration_count = 200 #2000
    step_scale = 0.3
    # tau = tau_scale / norm(a, 2)

    H = toeplitz_j(ref_td)
    tau = step_scale * shrinkage_factor(H)

    def soft_threshold(v):
        ret = np.zeros(v.shape)

        id_smaller = v <= -lambda_ * tau
        id_larger = v >= lambda_ * tau
        ret[id_smaller] = v[id_smaller] + lambda_ * tau
        ret[id_larger] = v[id_larger] - lambda_ * tau

        return ret

    def calc(H, sam_td, f):
        return np.dot(H.T, np.dot(H, f) - sam_td[:, 1])

    # init variables
    m = ref_td.shape[0]
    opt_sol = [np.zeros(m), 1, 0]
    relerr = 1
    n = 0

    f = np.zeros(m)
    ssq = norm(f, 1)
    big_f = []

    while (relerr > eps) and (n < max_iteration_count):
        n += 1
        pre_calc = calc(H, sam_td, f)
        f = soft_threshold(f - tau * pre_calc)

        big_f.append(0.5 * norm(sam_td[:, 1] - H @ f, 2) ** 2 + lambda_ * norm(f, 1))
        print(f"F(f): {big_f[-1]}")

        ssq_new = norm(f, 1)
        relerr = abs(1 - ssq / ssq_new)
        if not n % 100:
            print(f"relerr: {round(relerr, int(-np.log10(relerr)) + 2)}. Iteration: {n}\n")
        ssq = ssq_new

        opt_sol[0], opt_sol[1], opt_sol[2] = f, relerr, n

    print(f"Relerr of last iteration: {relerr}. Completed iterations: {n}")
    print(f"Err. of output solution: {opt_sol[1]} at iteration: {opt_sol[2]}\n")

    return opt_sol[0]


if __name__ == '__main__':
    measurement = OPMeasurement(area_idx=1)

    ref_td = measurement.get_ref(normalize=True, sub_offset=True)
    sam_metal_td = measurement.get_point(x=1.0, y=5.0, normalize=True, sub_offset=True)
    #sam_ceramic_td = measurement.get_point(x=3.5, y=5.0, normalize=True, sub_offset=True)
    sam_coating_td = measurement.get_point(x=7.0, y=5.0, normalize=True, sub_offset=True)

    f_metal = deconvolve(ref_td, sam_metal_td)
    #f_ceramic = deconvolve(ref_td, sam_ceramic_td)
    f_coating = deconvolve(ref_td, sam_coating_td)

    dt = measurement.info["dt"]
    #t_metal, t_ceramic, t_coating = np.argmax(f_metal)*dt, np.argmax(f_ceramic)*dt, np.argmax(f_coating)*dt
    t_metal, t_coating = np.argmax(f_metal) * dt, np.argmax(f_coating) * dt

    #peak_pos = f"Peak positions: ({t_metal}, {t_ceramic}, {t_coating}) ps"
    #thicknesses = "$d_{ceramic} = $" + f"{round((t_ceramic - t_metal) * c_thz / 2, 1)} um \n"
    peak_pos = f"Peak positions: ({t_metal}, {t_coating}) ps"
    #thicknesses += "$d_{ceramic} + d_{coating} = $" + f"{round((t_coating - t_metal) * c_thz / 2, 1)} um \n"
    thicknesses = "$d_{coating} = $" + f"{round((t_coating - t_metal) * c_thz / 2, 1)} um"

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    axs[0].plot(sam_metal_td[:, 0], sam_metal_td[:, 1], label="Metal x=1.0, y=5.0")
    #axs[0].plot(sam_ceramic_td[:, 0], sam_ceramic_td[:, 1], label="Ceramic x=3.5, y=5.0")
    axs[0].plot(sam_coating_td[:, 0], sam_coating_td[:, 1], label="Coating x=7.0, y=5.0")
    axs[0].set_xlim((0, 31))
    axs[0].set_xlabel("Time (ps)")
    axs[0].set_ylabel("Normalized amplitude")
    axs[0].legend()

    axs[1].text(12, 0.15, peak_pos)
    axs[1].text(12, 0.05, thicknesses)
    axs[1].plot(ref_td[:, 0], f_metal, label="Metal impulse response")
    #axs[1].plot(ref_td[:, 0], f_ceramic, label="Ceramic impulse response")
    axs[1].plot(ref_td[:, 0], f_coating, label="Coating impulse response")
    axs[1].set_xlim((0, 31))
    axs[1].set_xlabel("Delay (ps)")
    axs[1].set_ylabel("Normalized amplitude")
    axs[1].legend()

    # plt.savefig(plot_file_name, dpi=300)

    plt.show()
