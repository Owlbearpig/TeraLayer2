import pandas as pd
from consts import *
import numpy as np
import matplotlib.pyplot as plt


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

    set1_info = "240, 121, 0.05, 0.05, 1, 1400, 0.050, 30000"
    set2_info = "160, 201, 0.05, 0.05, 1, 1400, 0.050, 34000"

    def __init__(self, area_idx=0):
        if area_idx == 0:
            self.arr = np.load(self.files[0]).reshape((240, 121, 1400))
        else:
            self.arr = np.load(self.files[3]).reshape((160, 201, 1400))

    def image(self, type_="p2p"):
        if type_ == "p2p":
            img = np.max(np.abs(self.arr), axis=2)
        else:
            img = np.argmax(np.abs(self.arr), axis=2)
        plt.imshow(img[:, 50:150])


if __name__ == '__main__':
    measurement = OPMeasurement(area_idx=1)
    measurement.image(type_="p2p")
    plt.show()