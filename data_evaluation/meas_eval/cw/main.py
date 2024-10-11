import numpy as np

from consts import *
import pandas as pd
import matplotlib.pyplot as plt
from functions import do_ifft, filtering, do_fft


def pp(data_fd):
    m = data_fd.shape[0]
    zero_pad = 0 * m

    y = np.concatenate((data_fd[:m // 2, 1], np.zeros(zero_pad), data_fd[m // 2 + 1:, 1]))

    df = np.mean(np.diff(data_fd[:, 0]))
    last_f = data_fd[-1, 0]
    f = np.linspace(last_f + df, last_f + df * zero_pad, zero_pad - 1)
    f = np.append(data_fd[:, 0], f)

    # f = np.arange(0, len(y))

    data_fd = array([f, y]).T

    return data_fd


def unwrap(phi):
    phi = np.unwrap(phi)

    return phi


def to_cartesian(arr):
    phi_unwrap = np.unwrap(arr[:, 2])
    # phi = np.linspace(0, phi_unwrap[-1], len(arr[:, 2]))
    phi = arr[:, 2]

    data_cart = arr[:, 1] * np.exp(-1j * phi)

    data_fd = array([arr[:, 0] / MHz, data_cart]).T

    pos_range = (data_fd[:, 0] >= 0)
    pos_range[-1] = False

    data_fd = data_fd[pos_range, :]

    # data_fd = pp(data_fd)

    return data_fd


def average_sam():
    sam_cnt = len(list(hhi_data_dir.glob("**/*")))

    sam0_file = hhi_data_dir / f"Kopf_Ahmad_10x_0000"
    sam0_values = pd.read_csv(sam0_file).values

    avg_sam_values = sam0_values.copy()
    avg_sam_values[:, 1] = 0
    for sam_idx in range(sam_cnt):
        data_file = hhi_data_dir / f"Kopf_Ahmad_10x_{sam_idx:04}"
        avg_sam_values[:, 1] += pd.read_csv(data_file).values[:, 1]

    avg_sam_values[:, 1] /= sam_cnt

    avg_sam_fd = to_cartesian(avg_sam_values)

    return avg_sam_fd


def load_data(sam_idx=None, ret_bk_gnd=False, shift=0):
    data_dir = Path(ROOT_DIR / "data" / "T-Sweeper_and_TeraFlash" / "Lackierte Keramik" / "CW (T-Sweeper)")
    # data_dir = Path(ROOT_DIR / "matlab_enrique" / "Data")

    bk_gnd_file = data_dir / "BG_1000x_b.csv"
    # bk_gnd_file = data_dir / "BG_1000.csv"
    ref_file = data_dir / "ref_1000x_c.csv"
    # ref_file = data_dir / "ref_1000x.csv"

    if sam_idx is None:
        sam_idx = 0
    data_file = data_dir / "Kopf_Ahmad_3" / f"Kopf_Ahmad_10x_{sam_idx:04}"
    # data_file = data_dir / "Kopf_1x" / f"Kopf_1x_{sam_idx:04}"

    bk_gnd_fd = to_cartesian(pd.read_csv(bk_gnd_file).values)
    ref_fd = to_cartesian(pd.read_csv(ref_file).values)

    # sam_cnt = len(list((data_dir / "Kopf_Ahmad_3").glob("**/*")))

    sam_fd = to_cartesian(pd.read_csv(data_file).values)

    sam_fd[:, 1] -= bk_gnd_fd[:, 1]
    ref_fd[:, 1] -= bk_gnd_fd[:, 1]

    sam_fd[:, 1] *= np.exp(-1j*2*pi*sam_fd[:, 0].real*0.05*shift)  # probably not correct.

    if ret_bk_gnd:
        return ref_fd, sam_fd, bk_gnd_fd
    else:
        return ref_fd, sam_fd


def main():
    samples = [10]
    for sam_idx in samples:
        ref_fd, sam_fd, bk_gnd_fd = load_data(sam_idx=sam_idx, ret_bk_gnd=True, shift=1)

        t_func_fd = array([ref_fd[:, 0].real, sam_fd[:, 1] / ref_fd[:, 1]]).T
        # t_func_fd = pp(t_func_fd)

        avg_sam_fd = average_sam()

        ref_td, sam_td, bk_gnd_td = do_ifft(ref_fd), do_ifft(sam_fd), do_ifft(bk_gnd_fd)
        t_func_td = do_ifft(t_func_fd, shift=10)

        avg_sam_td = do_ifft(avg_sam_fd)

        # ref_td = filtering(ref_td, filt_type="bp", wn=(0.3, 1.7), order=5)
        # sam_td = filtering(sam_td, filt_type="bp", wn=(0.3, 1.7), order=5)
        t_func_td = filtering(t_func_td, filt_type="hp", wn=0.22, order=1)
        t_func_td = filtering(t_func_td, filt_type="lp", wn=1.72, order=4)
        
        t_func_fd_filtered = do_fft(t_func_td)
        
        freqs = sam_fd[:, 0].real

        df = np.mean(np.diff(freqs.real))
        dt = np.mean(np.diff(ref_td[:, 0].real))
        print(f"(CW) Frequency spacing: {df} THz, Sample period: {dt} ps")

        plt.figure("Time domain")
        plt.plot(ref_td[:, 0], ref_td[:, 1], label=f"Reference {sam_idx}")
        plt.plot(sam_td[:, 0], sam_td[:, 1], label=f"Sample {sam_idx}")
        plt.plot(t_func_td[:, 0], t_func_td[:, 1], label=f"Transfer function {sam_idx}")
        # plt.plot(avg_sam_td[:, 0], avg_sam_td[:, 1], label=f"Avg. Sample")
        # plt.plot(bk_gnd_td[:, 0], bk_gnd_td[:, 1], label=f"Background")
        # plt.plot(tmm_td[:, 0], tmm_td[:, 1], label=f"TMM * Reference")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

        plt.figure("Spectrum")
        plt.plot(ref_fd[:, 0], 20 * np.log10(np.abs(ref_fd[:, 1])), label=f"Reference")
        plt.plot(sam_fd[:, 0], 20 * np.log10(np.abs(sam_fd[:, 1])), label=f"Sample {sam_idx}")
        plt.plot(t_func_fd[:, 0], 20 * np.log10(np.abs(t_func_fd[:, 1])), label=f"Transfer function {sam_idx}")
        #plt.plot(t_func_fd[:, 0], 20 * np.log10(np.abs(t_func_fd_filtered[:, 1])), label=f"Transfer function filtered {sam_idx}")
        # plt.plot(avg_sam_fd[:, 0], 20 * np.log10(np.abs(avg_sam_fd[:, 1])), label=f"Avg. Sample")
        # plt.plot(sam_fd[:, 0], 20 * np.log10(np.abs(bk_gnd_fd[:, 1])), label=f"Background")
        # plt.plot(ref_fd[:, 0], 20 * np.log10(np.abs(r_tmm[:, 1] * ref_fd[:, 1])), label=f"TMM * Reference")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")
        plt.legend()

        plt.figure("Phase")
        #plt.plot(sam_fd[:, 0], np.angle(sam_fd[:, 1]), label="$\phi_{sam}$")
        #plt.plot(sam_fd[:, 0], np.angle(ref_fd[:, 1]), label="$\phi_{ref}$")
        plt.plot(t_func_fd[:, 0], np.angle(t_func_fd[:, 1]), label="$\phi_{Tfunc}$")
        #plt.plot(t_func_fd[:, 0], np.angle(t_func_fd_filtered[:, 1]), label="$\phi_{TfuncFilt}$")
        # plt.plot(avg_sam_fd[:, 0], np.angle(avg_sam_fd[:, 1]), label="$\phi_{Avg}$")
        # plt.plot(sam_fd[:, 0], np.angle(r_exp), label=r"$\phi_{sam} - \phi_{ref}$ " + f"{sam_idx}")
        # plt.plot(sam_fd[:, 0], np.angle(bk_gnd_fd[:, 1]), label=r"$\phi_{bkgnd}$")
        # plt.plot(sam_fd[:, 0], phase_tmm, label=r"$\phi_{TMM}$")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (Rad)")
        plt.legend()

        """
        plt.figure("Real part")
        plt.plot(sam_fd[:, 0], sam_fd[:, 1].real, label="sam")
        plt.plot(sam_fd[:, 0], ref_fd[:, 1].real, label="ref")
        #plt.plot(avg_sam_fd[:, 0], avg_sam_fd[:, 1].real, label="Avg")
        # plt.plot(sam_fd[:, 0], np.angle(r_exp), label=r"$\phi_{sam} - \phi_{ref}$ " + f"{sam_idx}")
        # plt.plot(sam_fd[:, 0], np.angle(bk_gnd_fd[:, 1]), label=r"$\phi_{bkgnd}$")
        # plt.plot(sam_fd[:, 0], phase_tmm, label=r"$\phi_{TMM}$")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Real part")
        plt.legend()

        plt.figure("Imag part")
        plt.plot(sam_fd[:, 0], sam_fd[:, 1].imag, label="sam")
        plt.plot(sam_fd[:, 0], ref_fd[:, 1].imag, label="ref")
        #plt.plot(avg_sam_fd[:, 0], avg_sam_fd[:, 1].imag, label="Avg")
        # plt.plot(sam_fd[:, 0], np.angle(r_exp), label=r"$\phi_{sam} - \phi_{ref}$ " + f"{sam_idx}")
        # plt.plot(sam_fd[:, 0], np.angle(bk_gnd_fd[:, 1]), label=r"$\phi_{bkgnd}$")
        # plt.plot(sam_fd[:, 0], phase_tmm, label=r"$\phi_{TMM}$")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Imag part")
        plt.legend()
        """

        plt.legend(loc='upper right')


if __name__ == '__main__':
    main()
    plt.show()
