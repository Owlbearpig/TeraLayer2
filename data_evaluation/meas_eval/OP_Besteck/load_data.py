import pandas as pd
from consts import *
import numpy as np
import matplotlib.pyplot as plt
from functions import filtering


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

        self.ref_td = np.load(self.files[1])


    def image(self, type_="p2p"):
        info = self.info
        if type_ == "p2p":
            grid_vals = np.max(np.abs(self.arr), axis=2)
        else:
            grid_vals = np.argmax(np.abs(self.arr), axis=2)

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


    def plot_point(self, x, y):
        dx, dy, dt = self.info["dx"], self.info["dy"], self.info["dt"]
        h = self.info["h"]

        y_td = self.arr[int(x/dx), h - int(y/dy)]
        t = np.arange(0, dt*len(y_td), dt)

        y_td = filtering(y_td, wn=(0.001, 9.999), filt_type="bandpass", order=5)

        plt.figure("Single point")
        if not self.plotted_ref:
            plt.plot(self.ref_td[:, 0]-self.ref_td[0, 0], self.ref_td[:, 1], label="Reference")
            self.plotted_ref = True

        plt.plot(t, y_td, label=f"x={x} (mm), y={y} (mm)")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (Arb. u.)")
        plt.legend()


if __name__ == '__main__':
    measurement = OPMeasurement(area_idx=1)
    measurement.image(type_="p2p")
    measurement.plot_point(x=1.0, y=5.0)
    measurement.plot_point(x=4.5, y=5.0)
    measurement.plot_point(x=7.0, y=5.0)
    plt.show()