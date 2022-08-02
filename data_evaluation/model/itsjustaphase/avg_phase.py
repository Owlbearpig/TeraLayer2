import numpy as np
from numpy import zeros
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
from scipy.signal import windows
import os

MHz = 10 ** 6
THz = 10 ** 12

if os.name == 'posix':
    dir_path = Path(r"/home/alex/PycharmProjects/TeraLayer2/data_evaluation/matlab_enrique/Data")
else:
    dir_path = Path(r"E:\Projects\TeraLayer2\data_evaluation\matlab_enrique\Data")

file_paths = [Path(root) / file for root, dirs, files in os.walk(dir_path) for file in files]

read_file = lambda file_path: pd.read_csv(file_path).values
bg_data, ref_data = read_file(file_paths[0]), read_file(file_paths[1])
data_array = np.array([read_file(file) for file in file_paths if "Kopf" in str(file)])

# np.unwrap = lambda x: x
# avg_phase = np.mean(np.unwrap(data_array[:, :, 2]), axis=0)

sam_idx = 55
freqs = data_array[0, :, 0] * MHz

plt.figure("Amplitude frequency domain")
plt.plot(freqs, 20*np.log10(np.abs(ref_data[:, 1])), label="reference")
plt.plot(freqs, 20*np.log10(np.abs(bg_data[:, 1])), label="background")
plt.plot(freqs, 20*np.log10(np.abs(data_array[sam_idx, :, 1])), label=f"sample idx: {sam_idx}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")

plt.figure("Phase frequency domain")
plt.plot(freqs, np.unwrap(ref_data[:, 2]), label=f"reference")
plt.plot(freqs, np.unwrap(data_array[sam_idx, :, 2]), label=f"sample idx: {sam_idx}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")
plt.legend()
plt.show()
