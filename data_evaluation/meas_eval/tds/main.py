import matplotlib.pyplot as plt
import numpy as np

from functions import do_fft, do_ifft, filtering
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


def load_data(sam_idx=None, signal_shift=0, ret_bk_gnd=False):
    data_dir = Path(ROOT_DIR / "data" / "T-Sweeper_and_TeraFlash" / "Lackierte Keramik" / "Puls (TeraFlash)")

    bk_gnd_file = data_dir / "2020_01_30_Background_1000pulses.csv"
    ref_file = data_dir / "2020_02_20_Metallplaetchen_Ref2_1000AVG.csv"
    data_file = data_dir / "2020_02_20_Ampelmaennchenkopf_Blickrichtung_rechts_1AVG.csv"

    bk_gnd_td = pd.read_csv(bk_gnd_file).values
    ref_td = pd.read_csv(ref_file).values
    sam_td = pd.read_csv(data_file).values

    sam_cnt = sam_td.shape[0]
    remove_dc = True

    if remove_dc:
        ref_td[:, 2] -= np.mean(ref_td[:, 2])

    t = ref_td[:, 0] - ref_td[0, 0]

    ref_td = np.array([t, ref_td[:, 2]]).T

    if remove_dc:
        for i in range(sam_cnt):
            sam_td[:, 1] -= np.mean(sam_td[:, 1])

    if sam_idx is not None:
        sam_td[sam_idx, :] = np.roll(sam_td[sam_idx, :], signal_shift)

        return ref_td, np.array([t, sam_td[sam_idx, :]]).T

    sam_td_avg = np.zeros_like(sam_td[0, :])
    for sam_idx in range(sam_cnt):
        sam_td_avg += np.roll(sam_td[sam_idx, :], signal_shift)
    sam_td_avg /= sam_cnt

    if remove_dc:
        sam_td_avg -= np.mean(sam_td_avg)

    if ret_bk_gnd:
        return ref_td, np.array([t, sam_td_avg]).T, np.array([t, bk_gnd_td[:, 1]]).T
    else:
        return ref_td, np.array([t, sam_td_avg]).T


def unwrap(data_fd, is_ref=True):
    freqs = data_fd[:, 0].real

    t0_ref = 8.50
    t0_sam = 8.76
    if is_ref:
        phi_0 = 2 * pi * freqs * t0_ref
    else:
        phi_0 = 2 * pi * freqs * t0_sam

    phase_unwrapped = np.unwrap(np.angle(data_fd[:, 1] * np.exp(-1j * phi_0)))

    return np.unwrap(np.angle(data_fd[:, 1]))


def cost():
    return 0


def main():
    samples = [None]
    for sam_idx in samples:
        ref_td, sam_td, bk_gnd_td = load_data(sam_idx=sam_idx, signal_shift=-5, ret_bk_gnd=True)

        # ref_td = filtering(ref_td, filt_type="hp", wn=2.3)
        # sam_td = filtering(sam_td, filt_type="hp", wn=2.3)

        sam_fd, ref_fd, bk_gnd_fd = do_fft(sam_td), do_fft(ref_td), do_fft(bk_gnd_td)

        freqs = sam_fd[:, 0].real

        d_list = [43.0, 641.0, 74.0]
        # d_list = [46.1, 619.4, 72.0]

        n = get_n(freqs, n_min=2.80, n_max=2.80)

        r_tmm = tmm_package_wrapper(freqs, d_list, n)
        r_tmm[:, 1] = r_tmm[:, 1] * -1

        phase_tmm = np.angle(r_tmm[:, 1])
        # phase_tmm = np.angle(r_tmm[:, 1])

        tmm_fd = array([r_tmm[:, 0].real, r_tmm[:, 1] * ref_fd[:, 1]]).T
        tmm_td = do_ifft(tmm_fd)

        print(np.argmax(ref_td[:, 1]) - np.argmax(sam_td[:, 1]))

        r_exp = sam_fd[:, 1] / ref_fd[:, 1]

        plt.figure("Time domain")
        plt.plot(ref_td[:, 0], ref_td[:, 1], label=f"Reference {sam_idx}")
        plt.plot(sam_td[:, 0], sam_td[:, 1], label=f"Sample {sam_idx}")
        plt.plot(bk_gnd_td[:, 0], bk_gnd_td[:, 1], label=f"Background")
        plt.plot(tmm_td[:, 0], tmm_td[:, 1], label=f"TMM * Reference")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

        plt.figure("Spectrum")
        plt.plot(ref_fd[:, 0], 20 * np.log10(np.abs(ref_fd[:, 1])), label=f"Reference {sam_idx}")
        plt.plot(sam_fd[:, 0], 20 * np.log10(np.abs(sam_fd[:, 1])), label=f"Sample {sam_idx}")
        plt.plot(sam_fd[:, 0], 20 * np.log10(np.abs(bk_gnd_fd[:, 1])), label=f"Background")
        plt.plot(ref_fd[:, 0], 20 * np.log10(np.abs(r_tmm[:, 1] * ref_fd[:, 1])), label=f"TMM * Reference")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")
        plt.legend()

        plt.figure("Phase")
        # plt.plot(sam_fd[:, 0], np.angle(sam_fd[:, 1]), label="$\phi_{sam}$")
        # plt.plot(sam_fd[:, 0], np.angle(ref_fd[:, 1]), label="$\phi_{ref}$")
        plt.plot(sam_fd[:, 0], np.angle(r_exp), label=r"$\phi_{sam} - \phi_{ref}$ " + f"{sam_idx}")
        plt.plot(sam_fd[:, 0], np.angle(bk_gnd_fd[:, 1]), label=r"$\phi_{bkgnd}$")
        plt.plot(sam_fd[:, 0], phase_tmm, label=r"$\phi_{TMM}$")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (Rad)")
        plt.legend()

        plt.legend(loc='upper right')


if __name__ == '__main__':
    main()
    plt.show()
