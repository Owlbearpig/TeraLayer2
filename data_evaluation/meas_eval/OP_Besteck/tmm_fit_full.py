import matplotlib.pyplot as plt
from load_data import OPMeasurement
from scipy.optimize import shgo
from consts import *
from numpy import array
from tmm import coh_tmm
from functions import do_ifft, do_fft, to_db, window
from mpl_settings import *


def shift_ref(ref_td, sam_td):
    dt = np.mean(np.diff(ref_td[:, 0].real))
    p1_idx0, p1_idx1 = int(17 / dt), int(20 / dt)
    p_ref_idx0, p_ref_idx1 = int(12 / dt), int(15 / dt)
    pos_pulse1_sam = p1_idx0 + np.argmax(sam_td[p1_idx0:p1_idx1, 1])
    pos_pulse1_ref = p_ref_idx0 + np.argmax(ref_td[p_ref_idx0:p_ref_idx1, 1])

    idx_shift = pos_pulse1_sam - pos_pulse1_ref

    shifted_ref_td = array([ref_td[:, 0], np.roll(ref_td[:, 1], idx_shift)]).T

    return shifted_ref_td


def tmm_eval(image, eval_point, en_plot=True):
    sam_td = image.get_point(*eval_point, sub_offset=True, both=False)
    ref_td = image.get_ref(normalize=False, sub_offset=False, both=False)

    ref_td = shift_ref(ref_td, sam_td)

    sam_td = window(sam_td, win_len=14, shift=0, en_plot=False, slope=0.15)
    #plt.show()
    ref_td = window(ref_td, win_len=14, shift=0, en_plot=False, slope=0.15)
    #plt.show()

    """
    plt.plot(ref_td[:, 0], ref_td[:, 1], label="ref")
    plt.plot(sam_td[:, 0], sam_td[:, 1], label="sam")
    plt.xlabel("Time (ps)")
    plt.ylabel("Amplitude (Arb. u.)")
    plt.legend()
    plt.show()
    """

    ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

    freqs = ref_fd[:, 0].real
    omega = 2 * pi * freqs
    one = np.ones_like(freqs)

    d_list = array([inf, 250, 1e10, inf])

    phase_shift = np.exp(-1j * 0 * omega / c_thz)

    def calc_model(n_list):
        rs_tmm_fd = np.zeros_like(freqs, dtype=complex)
        for f_idx, freq in enumerate(freqs):
            lam_vac = c_thz / freq
            n = n_list[f_idx]
            r_tmm_fd = -1 * coh_tmm("s", n, d_list, thea, lam_vac)["r"]
            rs_tmm_fd[f_idx] = r_tmm_fd

        sam_tmm_fd = array([freqs, rs_tmm_fd * ref_fd[:, 1] * phase_shift]).T
        sam_tmm_td = do_ifft(sam_tmm_fd)

        return sam_tmm_td, sam_tmm_fd

    bounds = [(1.8, 2.2), (0, 0.1)]
    n_metal = (500 + 1j*500) * one

    def cost(p, f_idx):
        n = array([one[f_idx], p[0] + 1j * p[1], n_metal[f_idx], one[f_idx]])
        lam_vac = c_thz / freqs[f_idx]
        r_tmm_fd = -1 * coh_tmm("s", n, d_list, thea, lam_vac)["r"]
        sam_tmm_fd = r_tmm_fd * ref_fd[f_idx, 1] * phase_shift[f_idx]

        amp_loss = (np.abs(sam_tmm_fd) - np.abs(sam_fd[f_idx, 1])) ** 2
        phi_loss = (np.angle(sam_tmm_fd) - np.angle(sam_fd[f_idx, 1])) ** 2

        return amp_loss + phi_loss

    shgo_iters = 7
    try:
        n_coat = np.load(f"n_coat_{d_list[1]}.npy")
    except FileNotFoundError:
        n_coat = np.zeros(len(freqs), dtype=complex)
        for f_idx, freq in enumerate(freqs):
            # lol while loop maybe ???
            if freq <= 2:
                print(f"Frequency: {freq} (THz), (idx: {f_idx})")
                if freq <= 0.25:
                    res = shgo(cost, bounds=bounds, args=(f_idx,), iters=5)
                else:
                    iters = shgo_iters
                    res = shgo(cost, bounds=bounds, args=(f_idx,), iters=iters - 2)
                    while res.fun > 1e-5:
                        iters += 1
                        res = shgo(cost, bounds=bounds, args=(f_idx,), iters=iters)
                        if iters >= 6:
                            break

                n_coat[f_idx] = res.x[0] + 1j * res.x[1]
                print(n_coat[f_idx], f"Fun: {res.fun}", "\n")
            else:
                n_coat[f_idx] = n_coat[f_idx-1]

        np.save(f"n_coat_{d_list[1]}.npy", n_coat)

    n_shgo = array([one, n_coat, n_metal, one]).T

    sam_tmm_shgo_td, sam_tmm_shgo_fd = calc_model(n_shgo)

    phi_tmm = np.angle(sam_tmm_shgo_fd[:, 1])

    label = f"(TMM) x={eval_point[0]} mm, y={eval_point[1]} mm"
    if en_plot:
        plt.figure("RI")
        plt.title("Refractive index")
        plt.plot(freqs[plot_range_OP], n_coat[plot_range_OP].real, label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Refractive index")

        plt.figure("Extinction coefficient")
        plt.title("Extinction coefficient")
        plt.plot(freqs[plot_range_OP], n_coat[plot_range_OP].imag, label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Extinction coefficient")

        plt.figure("Spectrum")
        plt.title("Spectrum")
        plt.plot(sam_tmm_shgo_fd[plot_range1, 0], to_db(sam_tmm_shgo_fd[plot_range1, 1]), label=label, color="Green")

        plt.figure("Phase")
        plt.title("Phases")
        plt.plot(sam_tmm_shgo_fd[plot_range_OP, 0], phi_tmm[plot_range_OP],
                 label=label, linewidth=1.5)

        plt.figure("Time domain")
        plt.plot(sam_tmm_shgo_td[:, 0], sam_tmm_shgo_td[:, 1], label=label, linewidth=2)

    print(f"{d_list[1]}", "Ri std: ", np.std(n_coat.real), "Extinction coeff. std: ", np.std(n_coat.imag))

    return n_shgo


if __name__ == '__main__':
    image = OPMeasurement(area_idx=1)
    eval_point = (7.0, 8.0)

    #eval_point = random.choice(image.all_points)
    # eval_point = (20, 10)  # used for s1-s3
    # eval_point = (33, 11)  # s4
    image.plot_point(*eval_point)
    tmm_eval(image, eval_point, en_plot=True)

    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        plt.legend()

    plt.show()

