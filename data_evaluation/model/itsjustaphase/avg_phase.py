import numpy as np
from numpy import zeros, pi
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from pathlib import Path
from scipy.signal import windows
import os

# mpl.rcParams['lines.linestyle'] = '--'
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2

MHz = 10 ** 6
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

# np.unwrap = lambda x: x
# avg_phase = np.mean(np.unwrap(data_array[:, :, 2]), axis=0)

sam_idx = 9
freqs = data_array[0, :, 0] * MHz

# f_slice = (0*THz < freqs) * (freqs < 1.69*THz)
f_slice = (0 * THz < freqs) * (freqs < 2.00 * THz)
#f_slice = (0.40 * THz < freqs) * (freqs < 0.50 * THz)
freqs = freqs[f_slice]


def phase_unwrap(arr):
    # unwrapping
    p_jumps = 1.1 * pi
    arr = np.unwrap(arr, discont=p_jumps)

    arr = np.diff(arr, append=arr[-1])

    """
    # remove single spikes
    for i in range(3, len(arr) - 3):
        if (abs(arr[i] - arr[i - 1]) > 1.2) * (abs(arr[i] - arr[i + 1]) > 1.2):
            arr[i] = (arr[i - 1] + arr[i + 1]) / 2
        # remove more "rounded" dips
        c1 = abs(arr[i - 2] - arr[i - 1]) > 2
        c2 = (arr[i - 1] > arr[i]) * (arr[i] < arr[i + 1])
        c3 = abs(arr[i + 1] - arr[i + 2]) > 2
        if c1*c2*c3:
            a = abs(arr[i - 2] - arr[i + 2]) / 4
            arr[i - 1], arr[i], arr[i+1] = a, 2*a, 3*a
    """

    return arr


plt.figure("Unwrapped phase frequency domain")
# diff = np.unwrap(ref_data[:, 2])[636]-np.unwrap(data_array[sam_idx, :, 2])[636]
diff = 0
p_uwrap_ref = np.unwrap(ref_data[f_slice, 2])
p_uwrap_sam = np.unwrap(data_array[sam_idx, f_slice, 2])
p_uwrap_mean = np.unwrap(np.mean(data_array[:, f_slice, 2], axis=0))
plt.plot(freqs, p_uwrap_ref, label=f"reference")
plt.plot(freqs, p_uwrap_mean, label=f"mean")
plt.plot(freqs, p_uwrap_sam, label=f"sample idx: {sam_idx}")
#plt.plot(freqs, phase_unwrap(data_array[sam_idx + 12, f_slice, 2]), label=f"sample idx: {sam_idx + 12}")
plt.plot(freqs,  p_uwrap_sam - p_uwrap_ref, label=f"Sam - ref, idx: {sam_idx}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")
plt.legend()

plt.figure("Phase diff frequency domain")
diff_phase = data_array[sam_idx + 1, f_slice, 2]-data_array[sam_idx, f_slice, 2]
plt.plot(freqs, diff_phase, label=f"sample idx: {sam_idx + 1} - {sam_idx}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")
plt.legend()

plt.figure("Raw phase frequency domain")
# plt.plot(freqs, ref_data[f_slice, 2], label=f"reference")
plt.plot(freqs, data_array[sam_idx, f_slice, 2], label=f"sample idx: {sam_idx}")
plt.plot(freqs, data_array[sam_idx + 1, f_slice, 2], label=f"sample idx: {sam_idx + 1}")
# plt.plot(freqs, data_array[sam_idx, f_slice, 2], label=f"sample idx: {sam_idx}")
# plt.plot(freqs, np.mean(data_array[:, f_slice, 2], axis=0), label=f"mean")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")
plt.legend()
plt.show()

Y_sam = data_array[sam_idx, f_slice, 1] * np.exp(1j * data_array[sam_idx, f_slice, 2])
Y_ref = ref_data[f_slice, 1] * np.exp(1j * ref_data[f_slice, 2] - 1j * diff)

y_sam = np.fft.ifft(Y_sam)
y_ref = np.fft.ifft(Y_ref)

plt.figure("Amplitude frequency domain")
plt.plot(freqs, 20 * np.log10(np.abs(ref_data[f_slice, 1])), label="reference")
plt.plot(freqs, 20 * np.log10(np.abs(bg_data[f_slice, 1])), label="background")
# plt.plot(freqs, 20*np.log10(np.abs(data_array[sam_idx, f_slice, 1])), label=f"sample idx: {sam_idx}")
plt.plot(freqs, 20 * np.log10(np.abs(np.mean(data_array[:, f_slice, 1], axis=0))), label=f"mean")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.legend()

plt.figure("Amplitude time domain")
plt.plot(y_ref, label=f"reference")
plt.plot(y_sam, label=f"sample idx: {sam_idx}")
plt.xlabel("Time (?)")
plt.ylabel("Amplitude (a.u.)")
plt.legend()
plt.show()
