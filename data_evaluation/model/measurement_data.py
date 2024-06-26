import numpy as np
from numpy import pi, array
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from helpers import is_iterable
import os

np.set_printoptions(suppress=True)

# mpl.rcParams['lines.linestyle'] = '--'
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2

MHz = 10 ** 6
GHz = 10 ** 9
THz = 10 ** 12

if os.name == 'posix':
    dir_path = Path(r"/home/alex/PycharmProjects/TeraLayer2/data_evaluation/matlab_enrique/Data")
else:
    dir_path = Path(r"E:\Projects\TeraLayer2\data_evaluation\matlab_enrique\Data")
    if not dir_path.exists():
        dir_path = Path(r"C:\Users\Laptop\PycharmProjects\TeraLayer2\data_evaluation\matlab_enrique\Data")

file_paths = [Path(root) / file for root, dirs, files in os.walk(dir_path) for file in files]

read_file = lambda file_path: pd.read_csv(file_path).values
bg_data, ref_data = read_file(file_paths[0]), read_file(file_paths[1])
data_array = np.array([read_file(file) for file in file_paths if "Kopf" in str(file)])


def get_measured_phase(freqs, sam_idx=slice(None)):
    if not is_iterable(freqs):
        freqs = [freqs]

    selected_freqs_idx = array([np.argwhere(np.isclose(freq, data_array[0, :, 0] * MHz))[0][0] for freq in freqs])
    return data_array[sam_idx, selected_freqs_idx, 2] - ref_data[selected_freqs_idx, 2]


def get_measured_amplitude(freqs, sam_idx=slice(None)):
    if not is_iterable(freqs):
        freqs = [freqs]

    selected_freqs_idx = array([np.argwhere(np.isclose(freq, data_array[0, :, 0] * MHz))[0][0] for freq in freqs])
    return np.real((data_array[sam_idx, selected_freqs_idx, 1] / ref_data[selected_freqs_idx, 1]) ** 2)


def get_bg_amplitude(freqs):
    if not is_iterable(freqs):
        freqs = [freqs]

    selected_freqs_idx = array([np.argwhere(np.isclose(freq, bg_data[:, 0] * MHz))[0][0] for freq in freqs])
    return np.real((bg_data[selected_freqs_idx, 1]) ** 2)


def get_bg_phase(freqs):
    if not is_iterable(freqs):
        freqs = [freqs]

    selected_freqs_idx = array([np.argwhere(np.isclose(freq, bg_data[:, 0] * MHz))[0][0] for freq in freqs])
    return bg_data[selected_freqs_idx, 2]


def get_ref_amplitude(freqs):
    if not is_iterable(freqs):
        freqs = [freqs]

    selected_freqs_idx = array([np.argwhere(np.isclose(freq, bg_data[:, 0] * MHz))[0][0] for freq in freqs])
    return np.real((ref_data[selected_freqs_idx, 1]) ** 2)


def get_ref_phase(freqs):
    if not is_iterable(freqs):
        freqs = [freqs]

    selected_freqs_idx = array([np.argwhere(np.isclose(freq, ref_data[:, 0] * MHz))[0][0] for freq in freqs])
    return ref_data[selected_freqs_idx, 2]


if __name__ == '__main__':
    sam_idx = 29
    freqs = data_array[0, :, 0] * MHz
    f_slice = (0.00 * THz <= freqs) * (freqs <= 1.55 * THz)

    freqs = freqs[f_slice]

    selected_freqs = array([0, 0.010, 1.087, 1.380]) * THz
    selected_freqs_idx = array([np.argwhere(np.isclose(freq, freqs))[0][0] for freq in selected_freqs])

    raw_phase = data_array[sam_idx, f_slice, 2] - ref_data[f_slice, 2]
    raw_phases = get_measured_phase(freqs)

    print(selected_freqs)
    print(get_measured_phase(selected_freqs, sam_idx))
    print(np.mean(get_measured_phase(0)))

    plt.figure("Std amplitude")
    plt.plot(freqs, np.std(get_measured_amplitude(freqs), axis=0), label=f"std(amplitude)")
    plt.plot(freqs, np.mean(get_measured_amplitude(freqs), axis=0), label=f"mean(amplitude)")
    plt.plot(freqs, get_measured_amplitude(freqs, sam_idx), label=f"sample: {sam_idx}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()

    plt.figure("Raw phase frequency domain")
    # plt.plot(freqs, ref_data[f_slice, 2], label=f"reference")
    # plt.plot(freqs, data_array[sam_idx, f_slice, 2], label=f"sample idx: {sam_idx}")
    # plt.plot(freqs, data_array[sam_idx + 1, f_slice, 2], label=f"sample idx: {sam_idx + 1}")
    plt.plot(freqs, np.std(raw_phases, axis=0), label=f"std(raw phases)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phases (rad)")
    plt.legend()

    phase_meas = get_measured_phase(selected_freqs)

    plt.figure(f"Single frequencies raw phase")
    for freq in selected_freqs:
        phase_meas = get_measured_phase(freq)

        plt.plot(np.arange(0, 101), phase_meas, label=f"Freq: {round(freq / THz, 3)}")

    plt.xlabel("Sample index")
    plt.ylabel("Phase (rad)")
    plt.ylim((-pi * 1.05, pi * 1.05))
    plt.legend()

    plt.figure(f"Background amplitude")

    phase_meas = get_measured_phase(freqs)

    plt.plot(freqs / THz, 20 * np.log10(bg_data[f_slice, 1]), label=f"Background")
    plt.plot(freqs / THz, 20 * np.log10(data_array[sam_idx, f_slice, 1]), label=f"Sample, idx: {sam_idx}")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude (dB)")
    plt.legend()

    plt.show()
