from consts import data_dir as data_root
from enum import Enum
import json
import pandas as pd
from datetime import datetime
from samples import SamplesEnum, Sample
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
from meas_eval.consts import c_thz, thea
from tmm_package import coh_tmm_slim_unsafe

data_dir = data_root / "Jumping Laser THz/Probe Measurements (Reflexion)/2024-01-11"

sub_dirs = ["Discrete Frequencies - WaveSource",
            "Discrete Frequencies - WaveSource (PIC-Freuqency Set)", "T-Sweeper",
            "Discrete Frequencies - PIC all sweeps",
            # "Discrete Frequencies - PIC",
            ]


class SystemEnum(Enum):
    PIC = 1
    WaveSource = 2
    WaveSourcePicFreq = 3
    TSweeper = 4
    Model = 5


class MeasTypeEnum(Enum):
    Background = 1
    Reference = 2
    Sample = 3


class Measurement:
    file_path = None
    freq = None
    freq_OSA = None
    name = None
    amp = None
    amp_avg = None
    phase = None
    phase_avg = None
    n_sweeps = None
    meas_type = None
    system = None
    timestamp = None
    sample = None
    r = None
    r_avg = None

    def __init__(self, file_path=None):
        self._parse_file(file_path)

    def __repr__(self):
        return f"({self.file_path.stem}, {self.timestamp}, {self.system.name})"

    def time_diff(self, meas):
        if self.system != SystemEnum.TSweeper:
            return self.timestamp - meas.timestamp
        else:
            return (self.timestamp - meas.timestamp).total_seconds()

    def _set_metadata(self):
        if "background" in str(self.file_path).lower() or "bkg" in str(self.file_path).lower():
            self.meas_type = MeasTypeEnum.Background
        elif "short" in str(self.file_path).lower():
            self.meas_type = MeasTypeEnum.Reference
        else:
            self.meas_type = MeasTypeEnum.Sample

        if "Discrete Frequencies - PIC" in str(self.file_path):
            self.system = SystemEnum.PIC
        elif ("Discrete Frequencies - WaveSource" in str(self.file_path) and
              ("(PIC-Freuqency Set)" not in str(self.file_path))):
            self.system = SystemEnum.WaveSource
        elif "Discrete Frequencies - WaveSource (PIC-Freuqency Set)" in str(self.file_path):
            self.system = SystemEnum.WaveSourcePicFreq
        else:
            self.system = SystemEnum.TSweeper

        if self.meas_type == MeasTypeEnum.Sample:
            file_path_str = str(self.file_path).lower()
            if "cube" in file_path_str:
                self.sample = SamplesEnum.blueCube
            elif "fp" in file_path_str and "probe2" in file_path_str:
                self.sample = SamplesEnum.fpSample2
            elif "fp" in file_path_str and "probe3" in file_path_str:
                self.sample = SamplesEnum.fpSample3
            elif "fp" in file_path_str and "probe5_plastic" in file_path_str:
                self.sample = SamplesEnum.fpSample5Plastic
            elif "fp" in file_path_str and "probe5_ceramic" in file_path_str:
                self.sample = SamplesEnum.fpSample5ceramic
            elif "fp" in file_path_str and "probe6" in file_path_str:
                self.sample = SamplesEnum.fpSample6
            elif "op_blue" in file_path_str and "pos1" in file_path_str:
                self.sample = SamplesEnum.opBluePos1
            elif "op_blue" in file_path_str and "pos2" in file_path_str:
                self.sample = SamplesEnum.opBluePos2
            elif "op_black" in file_path_str and "pos1" in file_path_str:
                self.sample = SamplesEnum.opBlackPos1
            elif "op_black" in file_path_str and "pos2" in file_path_str:
                self.sample = SamplesEnum.opBlackPos2
            elif "op_red" in file_path_str and "pos1" in file_path_str:
                self.sample = SamplesEnum.opRedPos1
            elif "op_red" in file_path_str and "pos2" in file_path_str:
                self.sample = SamplesEnum.opRedPos2
            elif "op_darkred" in file_path_str and "pos1" in file_path_str:
                self.sample = SamplesEnum.opDarkRedPos1
            elif "op_darkred" in file_path_str and "pos2" in file_path_str:
                self.sample = SamplesEnum.opDarkRedPos2
            elif "bw-ceramic" in file_path_str and "white" in file_path_str:
                self.sample = SamplesEnum.bwCeramicWhiteUp
            elif "bw-ceramic" in file_path_str and "black" in file_path_str:
                self.sample = SamplesEnum.bwCeramicBlackUp
            elif "ampelmann" in file_path_str and "right" in file_path_str:
                self.sample = SamplesEnum.ampelMannRight
            elif "ampelmann" in file_path_str and "left" in file_path_str:
                self.sample = SamplesEnum.ampelMannLeft
            elif "ampelmann" in file_path_str and "avg" in file_path_str:
                self.sample = SamplesEnum.ampelMannOld
            elif "op_tool" in file_path_str and "red_pos1" in file_path_str:
                self.sample = SamplesEnum.opToolRedPos1
            elif "op_tool" in file_path_str and "red_pos2" in file_path_str:
                self.sample = SamplesEnum.opToolRedPos2
            elif "op_tool" in file_path_str and "blue_pos1" in file_path_str:
                self.sample = SamplesEnum.opToolBluePos1
            elif "op_tool" in file_path_str and "blue_pos2" in file_path_str:
                self.sample = SamplesEnum.opToolBluePos2
        else:
            self.sample = SamplesEnum.empty

    def _parse_csv_file(self):
        with open(self.file_path, "r") as file:
            first_5_lines = [file.readline().strip() for _ in range(5)]

        self.timestamp = datetime.strptime(first_5_lines[1].split(" ")[-1], "%Y-%m-%dT%H:%M:%S.%fZ")

        pd_df = pd.read_csv(self.file_path, skiprows=4)
        self.freq = np.array(pd_df["Frequency (THz)"], dtype=float)

        amp, phase = pd_df["Amplitude Signal (a.u.)"], pd_df["Phase Signal (rad)"]

        self.amp = np.array(amp, dtype=float)
        self.phase = np.array(phase, dtype=float)
        self.n_sweeps = 1

    def _parse_json_file(self):
        def LoadData(fName='DataDict.json'):
            """
            Created on Thu Jan 11 14:23:21 2024

            @author: schwenson
            """

            with open(fName, 'r') as f:
                DataDict = json.load(f)

            # change all lists back to numpy arrays
            for key in DataDict.keys():
                if type(DataDict[key]) == list:
                    DataDict[key] = np.array(DataDict[key], dtype=float)

            return DataDict

        json_dict = LoadData(self.file_path)

        self.freq = json_dict["Frequency [THz]"]
        self.freq_OSA = json_dict["Frequency [THz] (OSA measurement)"]

        self.name = json_dict["Measurement"]
        self.n_sweeps = len(json_dict['Amplitude [A]'])
        self.timestamp = json_dict['measure #']

        sorted_indices = np.argsort(self.freq)

        self.freq = self.freq[sorted_indices]
        self.freq_OSA = self.freq_OSA[sorted_indices]  # ! Assume oder is the same

        phase = json_dict['Phase [rad]'][:, sorted_indices]
        phase_raw = json_dict["Phase [rad] (raw)"][:, sorted_indices]

        amp = json_dict['Amplitude [A]'][:, sorted_indices]
        raw_amp_key = [key for key in json_dict if ("Am" in key) and ("(raw)" in key)][0]
        amp_raw = json_dict[raw_amp_key][:, sorted_indices]

        phase *= np.sign(self.freq)
        phase_raw *= np.sign(self.freq)

        self.amp = np.array(amp, dtype=float)

        phase_sign = 1
        if self.system == SystemEnum.PIC:
            phase_sign = -1
        self.phase = phase_sign * np.array(phase, dtype=float)

    def _parse_file(self, file_path):
        self.file_path = file_path

        self._set_metadata()

        if ".json" in str(self.file_path):
            self._parse_json_file()
        elif ".csv" in str(self.file_path) and ("lock" not in str(self.file_path)):
            self._parse_csv_file()
        else:
            print(f"Can't parse {file_path.stem}")

        self._set_mean_vals()

    def _set_mean_vals(self):
        # average amp and phase over multiple consecutive measurements of same sample
        if self.n_sweeps == 1:
            self.amp_avg = self.amp
            self.phase_avg = self.phase
        else:
            self.amp_avg = np.mean(self.amp, axis=0)
            self.phase_avg = self._calculate_mean_phase()

    def _calculate_mean_phase(self, en_plot=False, use_kmeans=False):
        # fix jumps and calculate mean phase # consider using unwrapping
        phase_avg = np.zeros_like(self.freq)
        if np.isclose(self.phase, np.zeros_like(self.phase)).all():
            return np.zeros_like(self.freq)

        for freq_idx, freq in enumerate(self.freq):
            if not use_kmeans:
                phi_unwrapped = np.unwrap(self.phase[:, freq_idx])
                phase_avg[freq_idx] = np.mean(phi_unwrapped, axis=0)
            else:
                phi_1f = self.phase[:, freq_idx]
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                kmeans.fit(phi_1f.reshape(-1, 1))

                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                if np.abs(centers[0] - centers[1]) >= 2:
                    center0_cnt = len(phi_1f[np.abs(centers[0] - phi_1f) <= 1])
                    center1_cnt = len(phi_1f[np.abs(centers[1] - phi_1f) <= 1])
                    if center0_cnt > center1_cnt:
                        biggest_cluster = centers[0]
                    else:
                        biggest_cluster = centers[1]
                    phi_raw_avg_1f = biggest_cluster[0]
                else:
                    phi_raw_avg_1f = 0.5 * np.sum(centers)

                if en_plot:
                    plt.figure()
                    plt.title(self)
                    plt.plot(phi_1f)

                    plt.scatter(np.zeros_like(phi_1f), phi_1f, c=labels, s=30, cmap='viridis', alpha=0.5)
                    plt.scatter(np.zeros_like(centers), centers, marker='X', s=200, color='red', label='Centroids')

                    plt.legend()
                    plt.show()

                phase_avg[freq_idx] = phi_raw_avg_1f

        return phase_avg


class ModelMeasurement(Measurement):
    def __init__(self, sample_enum: SamplesEnum):
        ref_file = data_dir / "T-Sweeper" / "17_Gold_Plate_short.csv"
        super().__init__(ref_file)
        self.sample = sample_enum
        self.system = SystemEnum.Model
        self.meas_type = MeasTypeEnum.Sample
        self.name = f"Model {self.sample}"

    def simulate_sam_measurement(self, fast=False):
        err_conf = np.seterr(divide='ignore')
        has_iron_core = self.sample.value.has_iron_core

        n = self.sample.value.get_ref_idx(self.freq)

        d_truth = self.sample.value.thicknesses

        r_mod = np.zeros_like(self.freq, dtype=complex)
        for f_idx, freq in enumerate(self.freq):
            if fast and f_idx > 1600:
                continue

            if fast and (f_idx % 4) != 0:
                r_mod[f_idx] = r_mod[f_idx - 1]
                continue

            lam_vac = c_thz / freq
            if has_iron_core:
                d_ = np.array([np.inf, *d_truth, 10, np.inf], dtype=float)
            else:
                d_ = np.array([np.inf, *d_truth, np.inf], dtype=float)
            r_mod[f_idx] = -1 * coh_tmm_slim_unsafe("s", n[f_idx], d_, thea, lam_vac)

        for f_idx in range(len(self.freq)):
            if fast and (f_idx % 4) != 0:
                continue
                # r_mod[f_idx] = r_mod[f_idx - 1]

        ref_fd = np.array([self.freq, self.amp * np.exp(1j * self.phase)]).T

        self.amp, self.phase = np.abs(ref_fd[:, 1] * r_mod), np.angle(ref_fd[:, 1] * r_mod)
        self.amp_avg, self.phase_avg = self.amp, self.phase
        self.r, self.r_avg = r_mod, r_mod

        np.seterr(**err_conf)

        return n


def get_all_measurements(add_model_measurements=False):
    all_dirs = [dir_ for dir_ in [data_dir / sub_dir for sub_dir in sub_dirs]]
    all_files = [file_path for sublist in [list(dir_path.glob('*')) for dir_path in all_dirs] for file_path in sublist]

    excluded_files = ["01_Gold_Plate_short.csv"]
    included_filetypes = ["csv", "json"]

    filtered_files = []
    for file in all_files:
        if file.name in excluded_files:
            continue
        if file.suffix[1:] not in included_filetypes:
            continue

        filtered_files.append(file)

    all_measurements = [Measurement(file_path) for file_path in filtered_files]

    if add_model_measurements:
        for sample in SamplesEnum:
            all_measurements.append(ModelMeasurement(sample))

    return all_measurements


if __name__ == '__main__':
    for measurement in get_all_measurements():
        print(measurement)

    tmm = ModelMeasurement(SamplesEnum.fpSample2)
    print(tmm.sample)
