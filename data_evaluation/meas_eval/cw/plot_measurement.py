import matplotlib.pyplot as plt
import numpy as np
from model.cost_function import Cost
from functions import do_fft, do_ifft
from consts import *
import pandas as pd
from model.tmm_package import tmm_package_wrapper
from model.refractive_index import get_n
from RTL_sim.twos_compl_OF_v2 import real_data_cw


def load_data(sam_idx_=None, bk_gnd=False, polar=False):
    samples = 100
    data_dir = Path(ROOT_DIR / "data" / "T-Sweeper_and_TeraFlash" / "Lackierte Keramik" / "CW (T-Sweeper)")
    bk_gnd_file = data_dir / "BG_1000x_b.csv"
    ref_file = data_dir / "ref_1000x_c.csv"

    # data_dir = Path(ROOT_DIR / "matlab_enrique" / "Data")
    # bk_gnd_file = data_dir / "BG_1000.csv"
    # ref_file = data_dir / "ref_1000x.csv"

    f0_idx = 0  # 234
    f1_idx = 2436  # 4400

    bk_gnd_fd = array(pd.read_csv(bk_gnd_file).values, dtype=complex)[f0_idx:f1_idx, ]
    ref_fd = array(pd.read_csv(ref_file).values, dtype=complex)[f0_idx:f1_idx, ]

    sam_fd = np.zeros((samples + 1, *ref_fd.shape), dtype=complex)
    for sam_idx in range(samples + 1):
        data_file = data_dir / "Kopf_Ahmad_3" / f"Kopf_Ahmad_10x_{sam_idx:04}"
        # data_file = data_dir / "Kopf_1x" / f"Kopf_1x_{sam_idx:04}"
        sam_fd[sam_idx, :, :] = array(pd.read_csv(data_file).values)[f0_idx:f1_idx, ]
    # print(ref_td.shape)

    f_offset = 0.0296  # 0.0296 with screenshot works
    ref_fd[:, 0] = (ref_fd[:, 0] / 1e6 - f_offset)
    sam_fd[:, :, 0] = (sam_fd[:, :, 0] / 1e6 - f_offset)
    bk_gnd_fd[:, 0] = (bk_gnd_fd[:, 0] / 1e6 - f_offset)

    # ref_fd[:, 2] -= ref_fd[0, 2]
    if sam_idx_ is not None:
        sam_fd = sam_fd[sam_idx_,]
        # sam_fd[0, 2] -= sam_fd[0, 2]
    # else:
    #    sam_fd[:, 0, 2] -= sam_fd[:, 0, 2]
    # bk_gnd_fd[:, 2] -= bk_gnd_fd[0, 2]

    if polar:
        return ref_fd, sam_fd

    # ref_fd[:, 1] = np.abs(ref_fd[:, 1]) * np.exp(1j * (ref_fd[:, 2] - bk_gnd_fd[:, 2]))
    # sam_fd[:, 1] = np.abs(sam_fd[:, 1]) * np.exp(1j * (sam_fd[:, 2] - bk_gnd_fd[:, 2]))
    # amp_bk_gnd_fd = np.abs(bk_gnd_fd[:, 1])
    # amp_ref, amp_sam = np.abs(ref_fd[:, 1]) - amp_bk_gnd_fd, np.abs(sam_fd[:, 1]) - amp_bk_gnd_fd
    bk_gnd_fd[:, 1] = np.abs(bk_gnd_fd[:, 1]) * np.exp(-1j * bk_gnd_fd[:, 2])

    ref_fd[:, 1] = np.abs(ref_fd[:, 1] - bk_gnd_fd[:, 1]) * np.exp(-1j * ref_fd[:, 2])
    sam_fd[:, 1] = np.abs(sam_fd[:, 1] - bk_gnd_fd[:, 1]) * np.exp(-1j * sam_fd[:, 2])

    sam_fd = sam_fd[:, :2]
    ref_fd = ref_fd[:, :2]
    bk_gnd_fd = bk_gnd_fd[:, :2]

    if bk_gnd:
        return ref_fd, sam_fd, bk_gnd_fd

    return ref_fd, sam_fd


def load_phase(sam_idx_=None, bk_gnd=False):
    samples = 100
    data_dir = Path(ROOT_DIR / "data" / "T-Sweeper_and_TeraFlash" / "Lackierte Keramik" / "CW (T-Sweeper)")
    bk_gnd_file = data_dir / "BG_1000x_b.csv"
    ref_file = data_dir / "ref_1000x_c.csv"

    # data_dir = Path(ROOT_DIR / "matlab_enrique" / "Data")
    # bk_gnd_file = data_dir / "BG_1000.csv"
    # ref_file = data_dir / "ref_1000x.csv"

    f0_idx = 234
    f1_idx = 2000 + 500

    bk_gnd_fd = array(pd.read_csv(bk_gnd_file).values, dtype=complex)[f0_idx:f1_idx, ]
    ref_fd = array(pd.read_csv(ref_file).values, dtype=complex)[f0_idx:f1_idx, ]

    sam_fd = np.zeros((samples + 1, *ref_fd.shape), dtype=complex)
    for sam_idx in range(samples + 1):
        data_file = data_dir / "Kopf_Ahmad_3" / f"Kopf_Ahmad_10x_{sam_idx:04}"
        # data_file = data_dir / "Kopf_1x" / f"Kopf_1x_{sam_idx:04}"
        sam_fd[sam_idx, :, :] = array(pd.read_csv(data_file).values)[f0_idx:f1_idx, ]
    # print(ref_td.shape)

    ref_fd[:, 0] /= 1e6
    sam_fd[:, :, 0] /= 1e6

    if sam_idx_ is not None:
        sam_fd = sam_fd[sam_idx_,]

    ref_fd[:, 1] = np.abs(ref_fd[:, 1]) * np.exp(-1j * ref_fd[:, 2])
    sam_fd[:, 1] = np.abs(sam_fd[:, 1]) * np.exp(-1j * (sam_fd[:, 2] - bk_gnd_fd[:, 2]))
    bk_gnd_fd[:, 1] = np.abs(bk_gnd_fd[:, 1]) * np.exp(-1j * bk_gnd_fd[:, 2])

    sam_fd = sam_fd[:, :2]
    ref_fd = ref_fd[:, :2]
    bk_gnd_fd = bk_gnd_fd[:, :2]

    if bk_gnd:
        return ref_fd, sam_fd, bk_gnd_fd

    return ref_fd, sam_fd


def main():
    sam_idx = 12
    ref_fd, sam_fd = load_data()

    freqs = np.arange(0.000, 1.500 + 0.001, 0.001)
    meas_freqs = ref_fd[:, 0]

    p_sol = array([43.0, 641.0, 74.0])
    # p_sol = array([38.29, 600.44, 51.05])

    r_exp = Cost(freqs=freqs, p_solution=p_sol, noise_std_scale=0.00, plt_mod=False).r_exp

    R_meas = np.abs(sam_fd[sam_idx, :, 1]) / np.abs(ref_fd[:, 1])
    phi_meas = np.angle(sam_fd[sam_idx, :, 1] / ref_fd[:, 1])

    R0 = np.real(r_exp * np.conj(r_exp))
    phi0 = np.angle(r_exp)

    plt.figure("Amplitude")
    plt.plot(meas_freqs, 20 * np.log10(np.abs(ref_fd[:, 1])), label=f"reference {sam_idx:04}")
    plt.plot(meas_freqs, 20 * np.log10(np.abs(sam_fd[sam_idx, :, 1])), label=f"sample {sam_idx:04}")
    plt.plot(freqs, 20 * np.log10(R0), label=f"Model {p_sol}")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude (dB)")
    plt.legend()

    plt.figure("Phase")
    plt.plot(meas_freqs, ref_fd[:, 2], label=f"ref {sam_idx:04}")
    plt.plot(meas_freqs, sam_fd[sam_idx, :, 2], label=f"sam {sam_idx:04}")
    plt.plot(freqs, phi0, label=f"Model {p_sol}")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (Rad)")
    plt.legend()

    plt.figure("Amplitude reflectivity")
    plt.plot(meas_freqs, 20 * np.log10(R_meas), label=f"Measurement {sam_idx:04}")
    plt.plot(freqs, 20 * np.log10(R0), label=f"Model {p_sol}")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude (dB)")
    plt.legend()

    plt.figure("Phase reflectivity")
    plt.plot(meas_freqs, phi_meas, label=f"Measurement {sam_idx:04}")
    plt.plot(freqs, phi0, label=f"Model {p_sol}")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (Rad)")
    plt.legend()

    """
    plt.figure()
    # plt.plot(ref_fd[:, 0], 10 * np.log10(np.abs(ref_fd[:, 1])), label="reference")
    # plt.plot(sam_fd[:, 0], 10 * np.log10(np.abs(sam_fd[:, 1])), label="sample")
    plt.plot(sam_fd[:, 0], phase_diff, label="exp")
    plt.plot(sam_fd[:, 0], np.unwrap(np.angle(r_tmm[:, 1]))-pi, label="tmm")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (Rad)")
    plt.legend()
    """


if __name__ == '__main__':
    selected_measurements = [37, 62, 0, 10, 20, 30, 55, 60, 75]
    for sam_idx in selected_measurements:
        r_exp_meas = real_data_cw(sam_idx=sam_idx)
        plt.figure("r real part")
        plt.plot(r_exp_meas.real, label=f"{sam_idx:04}")
        plt.figure("r imag part")
        plt.plot(r_exp_meas.imag, label=f"{sam_idx:04}")

    plt.figure("r real part")
    plt.legend()
    plt.figure("r imag part")
    plt.legend()
    plt.show()

    main()
    plt.show()
