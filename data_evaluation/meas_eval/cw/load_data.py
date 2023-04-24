from consts import *
import pandas as pd
from functions import window, do_fft, do_ifft, shift


def raw_data(sam_idx_=None, bk_gnd=False, polar=False):
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


def processed_data(sam_idx_, ret_all=False):
    offset = 0

    ref_fd, sam_fd = raw_data(sam_idx_=sam_idx_)
    freqs = ref_fd[:, 0].real

    ref_td, sam_td = do_ifft(ref_fd), do_ifft(sam_fd)

    # ref_td, sam_td = filter(ref_td), filter(sam_td)

    ref_td, sam_td = shift(ref_td, 100 - offset), shift(sam_td, 100)

    # ref_td = window(ref_td, win_width=800, win_start=0, en_plot=False, slope=0.2, label="Ref")
    # sam_td = window(sam_td, win_width=800, win_start=0, en_plot=False, slope=0.2, label="Sam")

    ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

    t_func_fd = np.zeros_like(ref_fd, dtype=complex)
    t_func_fd[:, 0] = freqs
    t_func_fd[:, 1] = sam_fd[:, 1] / ref_fd[:, 1]

    if ret_all:
        return t_func_fd, ref_fd, sam_fd
    else:
        return t_func_fd


def mean_data(sam_idx_=None, ret_t_func=False):
    freqs_ = processed_data(0)[:, 0].real
    if sam_idx_ is None:
        try:
            t_func_fd_, ref_fd_, sam_fd_ = np.load("mean_data.npy")
            amp_meas_ = np.abs(t_func_fd_[:, 1])
            angle_meas_ = np.angle(t_func_fd_[:, 1])
        except FileNotFoundError:
            amp_meas_all, angle_meas_all, ref_all, sam_all = [], [], [], []
            for s_idx in range(101):
                t_func_fd_, ref_fd_, sam_fd_ = processed_data(s_idx, ret_all=True)

                """
                plt.figure("test")
                plt.plot(np.angle(t_func_fd), label="angle before")
                plt.figure("test2")
                plt.plot(np.abs(t_func_fd), label="magn before")
                """
                ref_all.append(ref_fd_[:, 1])
                sam_all.append(sam_fd_[:, 1])
                amp_meas_all.append(np.abs(t_func_fd_[:, 1]))
                angle_meas_all.append(np.angle(t_func_fd_[:, 1]))

            ref_all = array(ref_all, dtype=complex)
            sam_all = array(sam_all, dtype=complex)
            amp_meas_all = array(amp_meas_all, dtype=complex)
            angle_meas_all = array(angle_meas_all, dtype=complex)

            ref_fd_mean = array([freqs_, np.mean(ref_all, axis=0)]).T
            sam_fd_mean = array([freqs_, np.mean(sam_all, axis=0)]).T
            amp_meas_mean = np.mean(amp_meas_all, axis=0)
            angle_meas_mean = np.mean(angle_meas_all, axis=0)

            t_func_mean = array([freqs_, amp_meas_mean * np.exp(1j * angle_meas_mean)]).T

            data = array([t_func_mean, ref_fd_mean, sam_fd_mean], dtype=complex)

            np.save("mean_data.npy", data)

            ref_fd_, sam_fd_ = ref_fd_mean, sam_fd_mean
            t_func_fd_, amp_meas_, angle_meas_ = t_func_mean, amp_meas_mean, angle_meas_mean
    else:
        t_func_fd_, ref_fd_, sam_fd_ = processed_data(sam_idx_, ret_all=True)
        angle_meas_ = np.angle(t_func_fd_[:, 1])
        amp_meas_ = np.abs(t_func_fd_[:, 1])

    if ret_t_func:
        return t_func_fd_

    return t_func_fd_, ref_fd_, sam_fd_, amp_meas_, angle_meas_
