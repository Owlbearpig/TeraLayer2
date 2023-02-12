import matplotlib.pyplot as plt
import numpy as np

from functions import do_fft, do_ifft
from consts import *
import pandas as pd
from model.tmm_package import tmm_package_wrapper
from model.refractive_index import get_n
from RTL_sim.twos_compl_OF_v2 import real_data_cw


def load_data():
    samples = 100
    data_dir = Path(ROOT_DIR / "data" / "T-Sweeper_and_TeraFlash" / "Lackierte Keramik" / "CW (T-Sweeper)")

    bk_gnd_file = data_dir / "BG_1000x_b.csv"
    ref_file = data_dir / "ref_1000x_c.csv"

    bk_gnd_fd = pd.read_csv(bk_gnd_file).values
    ref_fd = pd.read_csv(ref_file).values

    sam_fd = np.zeros((samples+1, *ref_fd.shape))
    for sam_idx in range(samples + 1):
        data_file = data_dir / "Kopf_Ahmad_3" / f"Kopf_Ahmad_10x_{sam_idx:04}"
        sam_fd[sam_idx, :, :] = pd.read_csv(data_file).values

    # print(ref_td.shape)

    ref_fd[:, 0] /= 1e6
    sam_fd[:, :, 0] /= 1e6

    return ref_fd, sam_fd


def main():
    sam_idx = 0
    ref_fd, sam_fd = load_data()

    plt.figure()
    plt.plot(ref_fd[:, 0], 20 * np.log10(np.abs(ref_fd[:, 1])), label=f"reference {sam_idx:04}")
    sam_idx = 0
    plt.plot(sam_fd[sam_idx, :, 0], 20 * np.log10(np.abs(sam_fd[sam_idx, :, 1])), label=f"sample {sam_idx:04}")
    sam_idx = 12
    plt.plot(sam_fd[sam_idx, :, 0], 20 * np.log10(np.abs(sam_fd[sam_idx, :, 1])), label=f"sample {sam_idx:04}")
    sam_idx = 37
    plt.plot(sam_fd[sam_idx, :, 0], 20 * np.log10(np.abs(sam_fd[sam_idx, :, 1])), label=f"sample {sam_idx:04}")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude (dB)")
    plt.legend()

    sam_idx = 0
    plt.figure()
    plt.plot(ref_fd[:, 0], ref_fd[:, 2], label=f"ref {sam_idx:04}")
    plt.plot(sam_fd[sam_idx, :, 0], sam_fd[sam_idx, :, 2], label=f"sam {sam_idx:04}")
    sam_idx = 12
    plt.plot(sam_fd[sam_idx, :, 0], sam_fd[sam_idx, :, 2], label=f"sam {sam_idx:04}")
    sam_idx = 37
    plt.plot(sam_fd[sam_idx, :, 0], sam_fd[sam_idx, :, 2], label=f"sam {sam_idx:04}")
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
