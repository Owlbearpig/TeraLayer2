import pandas as pd
from consts import *
import numpy as np
import matplotlib.pyplot as plt
from functions import filtering, do_fft, window, unwrap

files = ["2023-02-17_Probe_blau_hinten_corrected.tim.csv",
         "2023-17-02_Referenz_n1.csv",
         "2023-17-02_Referenz_n1000.csv",
         "2023-22-02_Probe_blau_Metall_Lack_corrected.tim.csv"]


def save_as_npy():
    for file in files:
        try:
            arr = np.loadtxt(op_besteck_dir / file, skiprows=1, delimiter=",")
        except Exception:
            arr = np.loadtxt(op_besteck_dir / file, skiprows=3, delimiter=",")
        np.save(file.replace(".csv", ".npy"), arr)


class OPMeasurement:
    files = ["2023-02-17_Probe_blau_hinten_corrected.tim.npy",
             "2023-17-02_Referenz_n1.npy",
             "2023-17-02_Referenz_n1000.npy",
             "2023-22-02_Probe_blau_Metall_Lack_corrected.tim.npy"]
    files = [op_besteck_dir / file for file in files]

    # set1_info = "240, 121, 0.05, 0.05, 1, 1400, 0.050, 30000"
    set1_info = {"w": 240, "h": 121, "dx": 0.05, "dy": 0.05, "dt": 0.05, "samples": 1400}
    # set2_info = "160, 201, 0.05, 0.05, 1, 1400, 0.050, 34000"
    set2_info = {"w": 160, "h": 201, "dx": 0.05, "dy": 0.05, "dt": 0.05, "samples": 1400}
    plotted_ref = False

    geom = "r"

    def __init__(self, area_idx=0):
        self.area_idx = area_idx
        if self.area_idx == 0:
            self.info = self.set1_info
            self.arr = np.load(self.files[0]).reshape((self.info["w"], self.info["h"], self.info["samples"]))
        else:
            self.info = self.set2_info
            self.arr = np.load(self.files[3]).reshape((self.info["w"], self.info["h"], self.info["samples"]))
            self.arr = np.flip(self.arr, axis=0)

        self.ref_td = np.load(self.files[1])[0:self.info["samples"], :]

    def image(self, type_="p2p", extent=None):
        info = self.info
        if type_ == "p2p":
            grid_vals = np.max(np.abs(self.arr), axis=2)
        else:
            # grid_vals = np.argmax(np.abs(self.arr[:, :, int(17 / info["dt"]):int(20 / info["dt"])]), axis=2)
            arr = self.arr.copy()
            #arr[self.arr < 0] = 0
            grid_vals = array(int(17/info["dt"]) + np.argmax((arr[:, :, int(17/info["dt"]):int(20/info["dt"])]), axis=2), dtype=float)
            grid_vals *= info["dt"]
        if extent is None:
            extent = [0, info["w"] * info["dx"], 0, info["h"] * info["dy"]]

        w0, w1 = extent[:2]
        h0, h1 = extent[2:]

        grid_vals = grid_vals[
                    int(w0 / self.info["dx"]):int(w1 / self.info["dx"]),
                    int(h0 / self.info["dy"]):int(h1 / self.info["dy"])
                    ]

        fig = plt.figure(f"Image {type_}")
        ax = fig.add_subplot(111)
        ax.set_title(f"Area {self.area_idx}")
        fig.subplots_adjust(left=0.2)

        img = ax.imshow(grid_vals.transpose((1, 0)),
                        vmin=np.min(grid_vals), vmax=np.max(grid_vals),
                        origin="upper",
                        cmap=plt.get_cmap('jet'),
                        extent=extent)

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        cbar = fig.colorbar(img)
        cbar.set_label(r"ToF$_{p1}$ (ps)", rotation=270, labelpad=20)

    def get_ref(self, normalize=False, sub_offset=False, both=False):
        ref = self.ref_td.copy()

        if sub_offset:
            ref[:, 1] -= np.mean(ref[:, 1])

        if normalize:
            ref[:, 1] *= 1 / np.max(ref[:, 1])

        t = np.arange(0, self.info["dt"] * len(ref[:, 0]), self.info["dt"])

        ref_td = array([t, ref[:, 1]]).T

        if not both:
            return ref_td
        else:
            ref_fd = do_fft(ref_td)
            return ref_td, ref_fd

    def get_point(self, x, y, normalize=False, sub_offset=False, both=False):
        dx, dy, dt = self.info["dx"], self.info["dy"], self.info["dt"]
        h = self.info["h"]

        y_ = self.arr[int(x / dx), h - int(y / dy)]

        if sub_offset:
            y_ -= np.mean(y_)

        if normalize:
            y_ *= 1 / np.max(y_)

        y_td = np.array([np.arange(0, dt * len(y_), dt), y_]).T

        if not both:
            return y_td
        else:
            return y_td, do_fft(y_td)

    def plot_point(self, x, y, sam_td=None, sub_noise_floor=False, label="", td_scale=1):
        if sam_td is None:
            sam_td = self.get_point(x, y, sub_offset=True)
        ref_td = self.get_ref(sub_offset=True)
        # y_td = filtering(y_td, wn=(2.000, 3.000), filt_type="bandpass", order=5)

        sam_td = window(sam_td, win_len=14, shift=0, en_plot=False, slope=0.15)
        ref_td = window(ref_td, win_len=14, shift=0, en_plot=False, slope=0.15)

        ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

        # sam_td, sam_fd = phase_correction(sam_fd, fit_range=(0.55, 1.00), extrapolate=True, both=True)
        # ref_td, ref_fd = phase_correction(ref_fd, fit_range=(0.55, 1.00), extrapolate=True, both=True)

        if self.geom == "t":
            phi_ref, phi_sam = unwrap(ref_fd), unwrap(sam_fd)
        else:
            phi_ref, phi_sam = unwrap(ref_fd, only_ang=True), unwrap(sam_fd, only_ang=True)

        noise_floor = np.mean(20 * np.log10(np.abs(ref_fd[ref_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        if not self.plotted_ref:
            plt.figure("Spectrum")
            plt.plot(ref_fd[plot_range1, 0], 20 * np.log10(np.abs(ref_fd[plot_range1, 1])) - noise_floor,
                     label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")

            plt.figure("Phase")
            plt.plot(ref_fd[plot_range1, 0], phi_ref[plot_range1, 1], label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Phase (rad)")

            plt.figure("Time domain")
            plt.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
            plt.xlabel("Time (ps)")
            plt.ylabel("Amplitude (Arb. u.)")

            self.plotted_ref = True

        label += f" (x={x} (mm), y={y} (mm))"
        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        plt.figure("Spectrum")
        plt.plot(sam_fd[plot_range1, 0], 20 * np.log10(np.abs(sam_fd[plot_range1, 1])) - noise_floor, label=label)

        plt.figure("Phase")
        plt.plot(sam_fd[plot_range1, 0], phi_sam[plot_range1, 1], label=label)

        plt.figure("Time domain")
        plt.plot(sam_td[:, 0], td_scale * sam_td[:, 1], label=label + f" (Amplitude x {td_scale})")


if __name__ == '__main__':
    measurement = OPMeasurement(area_idx=1)
    measurement.image(type_="p2p")

    # area 0
    #measurement.plot_point(x=1.8, y=2.85)
    #measurement.plot_point(x=6.0, y=2.33)
    #measurement.plot_point(x=8.0, y=2.00)

    # area 1
    measurement.plot_point(x=1.0, y=4.0)
    #measurement.plot_point(x=3.5, y=5.0)
    measurement.plot_point(x=7.0, y=4.0)
    plt.show()
