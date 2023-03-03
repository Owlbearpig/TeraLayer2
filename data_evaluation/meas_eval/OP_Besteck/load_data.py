import pandas as pd
from consts import *
import numpy as np
import matplotlib.pyplot as plt
from functions import filtering, do_fft

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

    def image(self, type_="p2p"):
        info = self.info
        if type_ == "p2p":
            grid_vals = np.max(np.abs(self.arr), axis=2)
        else:
            grid_vals = np.argmax(np.abs(self.arr), axis=2)
        #"""
        grid_vals = grid_vals[
                    int(0 / self.info["dx"]):int(8 / self.info["dx"]),
                    int(2 / self.info["dy"]):int(8 / self.info["dy"])
                    ]
        #"""
        fig = plt.figure("Image")
        ax = fig.add_subplot(111)
        ax.set_title(f"Area {self.area_idx}")
        fig.subplots_adjust(left=0.2)
        extent = [0, info["w"] * info["dx"], 0, info["h"] * info["dy"]]
        img = ax.imshow(grid_vals.transpose((1, 0)),
                        vmin=np.min(grid_vals), vmax=np.max(grid_vals),
                        origin="upper",
                        cmap=plt.get_cmap('jet'),
                        extent=extent)

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        cbar = fig.colorbar(img)
        cbar.set_label(f"{type_}", rotation=270, labelpad=10)

    def get_ref(self, normalize=False, sub_offset=False):
        ref = self.ref_td.copy()

        if sub_offset:
            ref[:, 1] -= np.mean(ref[:, 1])

        if normalize:
            ref[:, 1] *= 1 / np.max(ref[:, 1])

        t = np.arange(0, self.info["dt"] * len(ref[:, 0]), self.info["dt"])

        return array([t, ref[:, 1]]).T

    def get_point(self, x, y, normalize=False, sub_offset=False):
        dx, dy, dt = self.info["dx"], self.info["dy"], self.info["dt"]
        h = self.info["h"]

        y_ = self.arr[int(x / dx), h - int(y / dy)]

        if sub_offset:
            y_ -= np.mean(y_)

        if normalize:
            y_ *= 1 / np.max(y_)

        y_td = np.array([np.arange(0, dt * len(y_), dt), y_]).T

        return y_td

    def plot_point(self, x, y):
        y_td = self.get_point(x, y)

        # y_td = filtering(y_td, wn=(2.000, 3.000), filt_type="bandpass", order=5)

        """
        plt.figure("Single point")
        if not self.plotted_ref:
            ref_td = self.get_ref()
            #plt.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
            self.plotted_ref = True
        """
        y_fd = do_fft(y_td)

        plt.figure("Spectrum")
        if not self.plotted_ref:
            ref_td = self.get_ref()
            ref_fd = do_fft(ref_td)
            plt.plot(ref_fd[:, 0], 20 * np.log10(np.abs(ref_fd[:, 1])), label="Reference")
            self.plotted_ref = True

        #plt.plot(y_td[:, 0], 20*np.log10(np.abs(y_fd[:, 1])), label=f"x={x} (mm), y={y} (mm)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")
        plt.legend()

        plt.figure("Time domain")
        plt.plot(y_td[:, 0], y_td[:, 1], label=f"x={x} (mm), y={y} (mm)")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (Arb. u.)")
        plt.legend()


if __name__ == '__main__':
    measurement = OPMeasurement(area_idx=1)
    measurement.image(type_="tof")

    # area 0
    #measurement.plot_point(x=1.8, y=2.85)
    #measurement.plot_point(x=6.0, y=2.33)
    #measurement.plot_point(x=8.0, y=2.00)

    # area 1
    measurement.plot_point(x=1.0, y=5.0)
    #measurement.plot_point(x=3.5, y=5.0)
    measurement.plot_point(x=7.0, y=8.0)
    plt.show()
