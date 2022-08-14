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
f_slice = (0.20 * THz < freqs) * (freqs < 1.75 * THz)
noise_1 = (0.870 * THz < freqs) * (freqs < 0.980 * THz)


"""
f_916 = np.argmin(np.abs(freqs-0.916*THz))
f_926 = np.argmin(np.abs(freqs-0.926*THz))

data_array[:, f_926, 2] = (data_array[:, f_926-1, 2] + data_array[:, f_926+1, 2]) / 2
data_array[:, f_916, 2] = data_array[:, f_916-1, 2]

plt.figure("Noise interval 1")
plt.plot(freqs[noise_1], data_array[sam_idx, noise_1, 2], label=f"{sam_idx}")
plt.scatter(freqs[noise_1], data_array[10, noise_1, 2], label=f"sam_idx {10}")
plt.scatter(freqs[noise_1], data_array[15, noise_1, 2], label=f"sam_idx {15}")
plt.scatter(freqs[noise_1], data_array[30, noise_1, 2], label=f"sam_idx {30}")
plt.scatter(freqs[noise_1], data_array[60, noise_1, 2], label=f"sam_idx {60}")
plt.scatter(freqs[noise_1], data_array[90, noise_1, 2], label=f"sam_idx {90}")
plt.plot(freqs[noise_1], np.mean(np.take(data_array[:, noise_1, 2], [10,15,30,60,90], axis=0), axis=0), label=f"mean")
plt.legend()
plt.show()
"""

freqs = freqs[f_slice]

def phase_fix(freqs, phase):
    p = phase.copy()
    noise_1 = (0.870 * THz < freqs) * (freqs < 0.980 * THz)

    for i in range(1, len(p[noise_1])-1):
        diff = p[noise_1][i] - p[noise_1][i - 1]
        if abs(diff) > 3:
            print(diff)
            p[noise_1] -= diff

            plt.figure()
            plt.plot(freqs[noise_1], p[noise_1])
            plt.show()

    return phase

def phase_unwrap(freqs, phase):
    # unwrapping
    f_slice0 = (0.908*THz < freqs)*(freqs < 0.936*THz)

    #phase[f_slice0] = np.mean(phase[f_slice0])

    phase = np.unwrap(phase)

    #arr = np.diff(arr, append=arr[-1])

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

    return phase


plt.figure("Unwrapped phase frequency domain")
# diff = np.unwrap(ref_data[:, 2])[636]-np.unwrap(data_array[sam_idx, :, 2])[636]
diff = 0
p_unwrap_ref = phase_unwrap(freqs, ref_data[f_slice, 2])
p_unwrap_sam = phase_unwrap(freqs, data_array[sam_idx, f_slice, 2])
p_unwrap_mean = phase_unwrap(freqs, np.mean(data_array[:, f_slice, 2], axis=0))
p_unwrap_diff = p_unwrap_sam - p_unwrap_ref
p_unwrap_diff_fixed = phase_fix(freqs, p_unwrap_diff)

plt.plot(freqs, p_unwrap_ref, label=f"reference")
plt.plot(freqs, p_unwrap_mean, label=f"mean")
plt.plot(freqs, p_unwrap_sam, label=f"sample idx: {sam_idx}")
#plt.plot(freqs, phase_unwrap(data_array[sam_idx + 12, f_slice, 2]), label=f"sample idx: {sam_idx + 12}")
plt.plot(freqs, p_unwrap_diff, label=f"Sam - ref, idx: {sam_idx}")
plt.plot(freqs, p_unwrap_diff_fixed, label=f"p_unwrap_diff_fixed")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")
plt.legend()

plt.figure("Phase diff frequency domain")
diff_phase = data_array[sam_idx + 1, f_slice, 2]-data_array[sam_idx, f_slice, 2]
#plt.plot(freqs, diff_phase, label=f"sample idx: {sam_idx + 1} - {sam_idx}")
plt.plot(freqs, np.diff(p_unwrap_diff, append=0), label=f"np.diff(sam - ref), idx: {sam_idx}")
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
