import matplotlib.pyplot as plt
from functions import do_fft, do_ifft
from consts import *
import pandas as pd
from model.tmm_package import tmm_package_wrapper
from model.refractive_index import get_n
from mpl_settings import *

"""
goal was to evaluate the refractive index as a function of frequency
but without reference measurement that's probably not possible.
"""


def pp(data_td):
    # TODO
    return


def load_data(sam_idx=0):
    data_dir = Path(ROOT_DIR / "data" / "T-Sweeper_and_TeraFlash" / "Lackierte Keramik" / "Puls (TeraFlash)")

    bk_gnd_file = data_dir / "2020_01_30_Background_1000pulses.csv"
    ref_file = data_dir / "2020_02_20_Metallplaetchen_Ref2_1000AVG.csv"
    data_file = data_dir / "2020_02_20_Ampelmaennchenkopf_Blickrichtung_rechts_1AVG.csv"

    bk_gnd_td = pd.read_csv(bk_gnd_file).values
    ref_td = pd.read_csv(ref_file).values
    sam_td = pd.read_csv(data_file).values
    # print(ref_td.shape)
    t = ref_td[:, 0] - ref_td[0, 0]

    ref_td = np.array([t, ref_td[:, 2]]).T

    return ref_td, np.array([t, sam_td[sam_idx, :]]).T


def unwrap(data_fd, is_ref=True):
    freqs = data_fd[:, 0].real

    t0_ref = 8.50
    t0_sam = 8.76
    if is_ref:
        phi_0 = 2 * pi * freqs * t0_ref
    else:
        phi_0 = 2 * pi * freqs * t0_sam

    phase_unwrapped = np.unwrap(np.angle(data_fd[:, 1] * np.exp(-1j * phi_0)))

    return -1 * np.unwrap(np.angle(data_fd[:, 1]))


def main():
    ref_td, sam_td = load_data(sam_idx=0)

    sam_fd = do_fft(sam_td)
    ref_fd = do_fft(ref_td)

    freqs = sam_fd[:, 0].real

    # d_list = [43.0, 641.0, 74.0]
    d_list = [46.1, 619.4, 72.0]
    # d_list = [0.0, 641.0, 0.0]

    n = get_n(freqs, n_min=2.80, n_max=2.80)

    r_tmm = tmm_package_wrapper(freqs, d_list, n)
    r_tmm[:, 1] = r_tmm[:, 1]  # * -1 * np.exp(-1j * 2*pi*freqs * 0.80)

    phase_tmm = np.angle(r_tmm[:, 1])
    # phase_tmm = np.angle(r_tmm[:, 1])

    tmm_fd = array([r_tmm[:, 0].real, r_tmm[:, 1] * ref_fd[:, 1]]).T
    tmm_td = do_ifft(tmm_fd)

    r_exp = sam_fd[:, 1] / ref_fd[:, 1]

    plt.figure()
    plt.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
    plt.plot(sam_td[:, 0], sam_td[:, 1], label="Sample")
    plt.plot(tmm_td[:, 0], tmm_td[:, 1], label="TMM * Reference")
    plt.xlabel("Time (ps)")
    plt.ylabel("Amplitude (nA)")
    plt.legend()

    plt.figure()
    plt.plot(ref_fd[:, 0], 20 * np.log10(np.abs(ref_fd[:, 1])), label="Reference")
    plt.plot(sam_fd[:, 0], 20 * np.log10(np.abs(sam_fd[:, 1])), label="Sample")
    plt.plot(ref_fd[:, 0], 20 * np.log10(np.abs(r_tmm[:, 1] * ref_fd[:, 1])), label="TMM * Reference")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude (dB)")
    plt.legend()
    """
    plt.figure()
    # plt.plot(ref_fd[:, 0], 10 * np.log10(np.abs(ref_fd[:, 1])), label="reference")
    # plt.plot(sam_fd[:, 0], 10 * np.log10(np.abs(sam_fd[:, 1])), label="sample")
    plt.plot(sam_fd[:, 0], np.abs(r_exp), label="r_exp")
    plt.plot(sam_fd[:, 0], np.abs(r_tmm[:, 1]), label="r_tmm")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude (dB)")
    plt.legend()
    """
    plt.figure()
    # plt.plot(ref_fd[:, 0], 10 * np.log10(np.abs(ref_fd[:, 1])), label="reference")
    # plt.plot(sam_fd[:, 0], 10 * np.log10(np.abs(sam_fd[:, 1])), label="sample")
    plt.plot(sam_fd[:, 0], np.angle(r_exp), label="$\phi_{sam} - \phi_{ref}$")
    plt.plot(sam_fd[:, 0], phase_tmm, label="$\phi_{TMM}$")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (Rad)")
    plt.legend()

    ref_phase = unwrap(ref_fd)
    sam_phase = unwrap(sam_fd)
    phase_diff = sam_phase - ref_phase
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
    plt.legend(loc='upper right')

if __name__ == '__main__':
    main()
    plt.show()
