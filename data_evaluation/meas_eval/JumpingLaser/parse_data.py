from consts import data_dir as data_root
from enum import Enum
import json
import pandas as pd
from datetime import datetime
from samples import *

data_dir = data_root / "Jumping Laser THz/Probe Measurements (Reflexion)/2024-01-11"

sub_dirs = ["Discrete Frequencies - PIC", "Discrete Frequencies - WaveSource",
            "Discrete Frequencies - WaveSource (PIC-Freuqency Set)", "T-Sweeper"]


excluded_files = ["01_Gold_Plate_short.csv"]

class SystemEnum(Enum):
    PIC = 1
    WaveSource = 2
    WaveSourcePicFreq = 3
    TSweeper = 4


class MeasTypeEnum(Enum):
    Background = 1
    Reference = 2
    Sample = 3
    NotAMeasurement = 4


class Measurement:
    file_path = None
    freq = None
    freq_OSA = None
    name = None
    amp = None
    amp_avg = None
    phase = None
    phase_avg = None
    data_car = None
    data_car_avg = None
    n_sweeps = None
    meas_type = None
    system = None
    timestamp = None
    sample = None
    r_exp_car = None
    r_exp_car_avg = None

    def __init__(self, file_path):
        self.parse_file(file_path)
        if self.meas_type != MeasTypeEnum.NotAMeasurement:
            self.set_car_data()

    def __repr__(self):
        return f"({self.file_path.stem}, {self.timestamp}, {self.system})"

    def time_diff(self, meas):
        if self.system != SystemEnum.TSweeper:
            return self.timestamp - meas.timestamp
        else:
            return (self.timestamp - meas.timestamp).total_seconds()

    def read_metadata(self):
        if self.meas_type == MeasTypeEnum.NotAMeasurement:
            return

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
                self.sample = blue_cube
            elif "fp" in file_path_str and "probe2" in file_path_str:
                self.sample = fpSample2
            elif "fp" in file_path_str and "probe3" in file_path_str:
                self.sample = fpSample3
            elif "fp" in file_path_str and "probe5_plastic" in file_path_str:
                self.sample = fpSample5Plastic
            elif "fp" in file_path_str and "probe5_ceramic" in file_path_str:
                self.sample = fpSample5ceramic
            elif "fp" in file_path_str and "probe6" in file_path_str:
                self.sample = fpSample6
            elif "op_blue" in file_path_str and "pos1" in file_path_str:
                self.sample = opBluePos1
            elif "op_blue" in file_path_str and "pos2" in file_path_str:
                self.sample = opBluePos2
            elif "op_black" in file_path_str and "pos1" in file_path_str:
                self.sample = opBlackPos1
            elif "op_black" in file_path_str and "pos2" in file_path_str:
                self.sample = opBlackPos2
            elif "op_red" in file_path_str and "pos1" in file_path_str:
                self.sample = opRedPos1
            elif "op_red" in file_path_str and "pos2" in file_path_str:
                self.sample = opRedPos2
            elif "op_darkred" in file_path_str and "pos1" in file_path_str:
                self.sample = opDarkRedPos1
            elif "op_darkred" in file_path_str and "pos2" in file_path_str:
                self.sample = opDarkRedPos2
            elif "bw-ceramic" in file_path_str and "white" in file_path_str:
                self.sample = bwCeramicWhiteUp
            elif "bw-ceramic" in file_path_str and "black" in file_path_str:
                self.sample = bwCeramicBlackUp
            elif "ampelmann" in file_path_str and "right" in file_path_str:
                self.sample = ampelMannRight
            elif "ampelmann" in file_path_str and "left" in file_path_str:
                self.sample = ampelMannLeft
            elif "ampelmann" in file_path_str and "avg" in file_path_str:
                self.sample = ampelMannLeft
            elif "op_tool" in file_path_str and "red_pos1" in file_path_str:
                self.sample = opToolRedPos1
            elif "op_tool" in file_path_str and "red_pos2" in file_path_str:
                self.sample = opToolRedPos2
            elif "op_tool" in file_path_str and "blue_pos1" in file_path_str:
                self.sample = opToolBluePos1
            elif "op_tool" in file_path_str and "blue_pos2" in file_path_str:
                self.sample = opToolBluePos2
        else:
            self.sample = empty

    def parse_csv_file(self):
        with open(self.file_path, "r") as file:
            first_5_lines = [file.readline().strip() for _ in range(5)]

        self.timestamp = datetime.strptime(first_5_lines[1].split(" ")[-1], "%Y-%m-%dT%H:%M:%S.%fZ")

        pd_df = pd.read_csv(self.file_path, skiprows=4)
        self.freq = np.array(pd_df["Frequency (THz)"], dtype=float)

        amp, phase = pd_df["Amplitude Signal (a.u.)"], pd_df["Phase Signal (rad)"]

        self.amp = np.array(amp, dtype=float)
        self.phase = np.array(phase, dtype=float)
        self.n_sweeps = 1

        self.amp_avg = self.amp
        self.phase_avg = self.phase

    def parse_json_file(self):
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

        phase = json_dict['Phase [rad]'] * np.sign(self.freq)
        # self.freq = np.abs(self.freq)
        # self.freq_OSA = np.abs(self.freq_OSA)

        amp = json_dict['Amplitude [A]']

        self.amp = np.array(amp, dtype=float)
        self.phase = np.array(phase, dtype=float)

        self.amp_avg = np.mean(self.amp, axis=0)
        self.phase_avg = np.mean(self.phase, axis=0)

    def set_car_data(self):
        self.data_car = self.amp * np.exp(-1j * self.phase)  # ?? sign
        self.data_car_avg = self.amp_avg * np.exp(-1j * self.phase_avg)  # ?? sign

    def parse_file(self, file_path):
        self.file_path = file_path

        if ".json" in str(self.file_path):
            self.parse_json_file()
        elif ".csv" in str(self.file_path) and ("lock" not in str(self.file_path)):
            self.parse_csv_file()
        else:
            self.meas_type = MeasTypeEnum.NotAMeasurement

        self.read_metadata()


def get_all_measurements(ret_all_files=False):

    all_dirs = [dir_ for dir_ in [data_dir / sub_dir for sub_dir in sub_dirs]]
    all_files = [file_path for sublist in [list(dir_path.glob('*')) for dir_path in all_dirs] for file_path in sublist]

    all_files = [file for file in all_files if file.name not in excluded_files]

    all_measurements_ret = [Measurement(file_path) for file_path in all_files]

    if ret_all_files:
        return all_measurements_ret
    else:
        return [meas for meas in all_measurements_ret if (meas.meas_type != MeasTypeEnum.NotAMeasurement)]


if __name__ == '__main__':
    for measurement in get_all_measurements():
        print(measurement)
