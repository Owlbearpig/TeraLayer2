import numpy as np
from numpy import zeros, pi, array
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

# np.unwrap = lambda x: x
# avg_phase = np.mean(np.unwrap(data_array[:, :, 2]), axis=0)

sam_idx = 28
freqs = data_array[0, :, 0] * MHz
freqs_raw = freqs.copy()

# f_slice = (0*THz < freqs) * (freqs < 1.69*THz)
f_slice = (0.00 * THz <= freqs) * (freqs <= 1.85 * THz)
noise_1 = (0.870 * THz <= freqs) * (freqs <= 0.980 * THz)

freqs = freqs[f_slice]

selected_freqs = array([0.365, 0.503, 0.520, 1.087, 1.298, 1.380]) * THz
selected_freqs_idx = array([np.argwhere(np.isclose(freq, freqs))[0][0] for freq in selected_freqs])

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

raw_phase = data_array[sam_idx, f_slice, 2] - ref_data[f_slice, 2]
print(freqs[selected_freqs_idx])
print(raw_phase[selected_freqs_idx])
print(np.mean(data_array[:, f_slice, 2], axis=0)[selected_freqs_idx])
#np.save(f"freqs_measured_selected_idx_{sam_idx}", freqs[selected_freqs_idx])
#np.save(f"phase_measured_selected_idx_{sam_idx}", raw_phase[selected_freqs_idx])


def raw_phase_fix(freqs, phase, label_ext=""):
    p = phase.copy()

    for i in range(1, len(p)):
        diff = p[i-1]-p[i]
        if np.abs(diff) > pi:
            p[i] = p[i] + np.sign(diff)*2*pi

    plt.plot(freqs, p, label="raw_phase_fix" + label_ext)

    return p

def diff_quotient(freqs, p):
    h = 10
    diff_q = []
    for i in range(0, len(p)-h):
        diff_q.append((p[i+h] - p[i]) / h)

    return freqs[:len(p)-h], np.abs(np.array(diff_q))


def phase_interpol(freqs, p, skip_range):
    interval = (skip_range[0] <= freqs) * (freqs <= skip_range[1])
    pre_interval = np.roll(interval, -np.sum(interval))

    a, b = np.mean(np.diff(p[pre_interval])), p[interval][0]
    p[interval] = a * np.arange(0, np.sum(interval)) + b
    shift = p[interval][-1] - p[np.roll(interval, 1)][-1]

    p[np.argwhere(interval)[-1, 0] + 1:] += shift

    return p



def phase_fix(freqs, phase):
    p = phase.copy()
    start_idx = 600 - 200

    #p = phase_interpol(freqs, p, (0.910 * THz, 0.930 * THz))
    #p = phase_interpol(freqs, p, (1.136 * THz, 1.142 * THz))
    #p = phase_interpol(freqs, p, (1.411 * THz, 1.434 * THz))

    done = False
    jumps_found = []
    while not done:
        #diffs = np.diff(p, append=p[-1])
        _, diffs = diff_quotient(freqs, p)
        for idx, diff in enumerate(diffs):
            if (np.abs(diff) >= 0.50) and (idx >= start_idx)*(idx <= len(diffs)-10):
                f_start, f_end = freqs[idx - 5], freqs[idx + 15]
                p = phase_interpol(freqs, p, (f_start, f_end))
                jumps_found.append(idx)
                break
            if idx == len(diffs)-1:
                done = True

    print(f"Found jumps @: {jumps_found}, \nFreqs {np.round(freqs[array(jumps_found, dtype=int)] / THz, 3)}")

    return p


def phase_unwrap(phase, **kwargs):
    phase = np.unwrap(phase, **kwargs)

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
p_unwrap_ref = phase_unwrap(ref_data[f_slice, 2])
p_unwrap_sam = phase_unwrap(data_array[sam_idx, f_slice, 2])
p_unwrap_mean = phase_unwrap(np.mean(data_array[:, f_slice, 2], axis=0))
p_unwrap_diff = p_unwrap_sam - p_unwrap_ref
p_unwrap_diff_fixed = phase_fix(freqs, p_unwrap_diff)

plt.plot(freqs, p_unwrap_ref, label=f"reference")
plt.plot(freqs, p_unwrap_mean, label=f"mean")
plt.plot(freqs, p_unwrap_sam, label=f"sample idx: {sam_idx}")
# plt.plot(freqs, phase_unwrap(data_array[sam_idx + 12, f_slice, 2]), label=f"sample idx: {sam_idx + 12}")
plt.plot(freqs, p_unwrap_diff, label=f"Sam - ref, idx: {sam_idx}")
plt.plot(freqs, p_unwrap_diff_fixed, label=f"p_unwrap_diff_fixed")
# plt.plot(freqs, phase_fix(freqs, p_unwrap_ref), label=f"p_unwrap_ref_fixed")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")
plt.legend()

plt.figure("Mean of unwrapped fixed phases")
unwrapped_phases = []
for i in range(0, 1):
    p_unwrap_ref = phase_unwrap(ref_data[f_slice, 2])
    p_unwrap_sam = phase_unwrap(data_array[i, f_slice, 2])
    p_unwrap_diff = p_unwrap_sam - p_unwrap_ref
    unwrapped_phases.append(phase_fix(freqs, p_unwrap_diff))
    if i == 46:
        plt.plot(freqs, phase_fix(freqs, p_unwrap_diff), label=f"sam idx {i} of p_unwrap_diff_fixed")

plt.plot(freqs, np.mean(array(unwrapped_phases), axis=0), label=f"Mean of p_unwrap_diff_fixed")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")
plt.legend()

#np.save("freqs_measured_mean", freqs)
#np.save("phase_measured_mean", np.mean(array(unwrapped_phases), axis=0))

plt.figure("Phase diff frequency domain")
diff_phase = data_array[sam_idx + 1, f_slice, 2] - data_array[sam_idx, f_slice, 2]
# plt.plot(freqs, diff_phase, label=f"sample idx: {sam_idx + 1} - {sam_idx}")
#plt.plot(freqs, np.abs(np.diff(p_unwrap_diff, append=p_unwrap_diff[-1])), label=f"np.diff(sam - ref), idx: {sam_idx}")
plt.plot(*diff_quotient(freqs, p_unwrap_diff), label=f"diff_quotient, idx: {sam_idx}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")
plt.legend()

f_sam_idx = 1526 + 234
# print(data_array[sam_idx, f_slice, 2][1400])
print(freqs_raw[f_sam_idx] / THz)
print(np.mean(data_array[:, f_sam_idx, 2]))
plt.figure(f"Single frequency raw phase {round(freqs_raw[f_sam_idx] / THz, 3)}")
plt.plot(np.arange(0, 101), data_array[:, f_sam_idx, 2])
plt.xlabel("Sample index")
plt.ylabel("Phase (rad)")
plt.ylim((-pi * 1.05, pi * 1.05))
plt.legend()

plt.figure("Fixed raw phase frequency domain")
raw_phase_fix(freqs, raw_phase)
plt.plot(freqs, ref_data[f_slice, 2], label=f"reference")
plt.plot(freqs, data_array[sam_idx, f_slice, 2], label=f"sample idx: {sam_idx}")
#plt.plot(freqs, data_array[sam_idx + 1, f_slice, 2], label=f"sample idx: {sam_idx + 1}")
plt.plot(freqs, raw_phase, label=f"sam - ref, idx: {sam_idx}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")
plt.legend()

plt.figure("Raw phase frequency domain")
plt.plot(freqs, ref_data[f_slice, 2], label=f"reference")
plt.plot(freqs, data_array[sam_idx, f_slice, 2], label=f"sample idx: {sam_idx}")
#plt.plot(freqs, data_array[sam_idx, f_slice, 2], label=f"sample idx: {sam_idx}")
#plt.plot(freqs, data_array[sam_idx + 1, f_slice, 2], label=f"sample idx: {sam_idx + 1}")
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
