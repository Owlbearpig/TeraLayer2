# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:00:13 2021

@author: Talebf
"""
from scipy import signal
from scipy.constants import c
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from scipy.stats import norm

mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['axes.grid'] = True
# print(mpl.rcParams.keys())

THz = 10 ** 12


def fix_discontinuities(p_raw):
    for i in range(1, len(p_raw)-1):
        if np.abs(p_raw[i-1] - p_raw[i]) > 5:
            p_raw[i] = p_raw[i] + 2*pi * np.sign(p_raw[i-1] - p_raw[i])
    return p_raw

def plot_freq_response(b, a, fs, worN=8000):
    # Plot the freqsuency response.
    w, h = signal.freqz(b, a, worN=worN)
    plt.figure("Filter Frequency Response")
    # plt.plot(0.5 * fs * w / (10 ** 12 * np.pi), 20 * np.log10(np.abs(h)), 'b')
    plt.plot(0.5 * fs * w / np.pi, 20 * np.log10(np.abs(h)))
    # plt.plot(highcut, 0.5*np.sqrt(2), 'ko')
    # plt.axvline(highcut, color='k')
    # plt.xlim(0, 0.5*fs / GHz)
    plt.title("Filter Frequency Response")
    plt.xlabel('Frequency (THz)')
    plt.grid()


def butter_highpass(data, lowcut, fs, order=7, plot=False):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='high')

    f = signal.filtfilt(b, a, data)
    if plot:
        plot_freq_response(b, a, fs)

    return f


def butter_lowpass(data, highcut, fs, order=7, plot=False):
    # nyq = 0.5 * fs
    # high = highcut / nyq
    # print(highcut, nyq)
    b, a = signal.butter(order, highcut, btype='low', fs=fs)

    f = signal.lfilter(b, a, data)
    if plot:
        plot_freq_response(b, a, fs)

    return f


def butter_bandpass(data, lowcut, highcut, fs, order=7, plot=False):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    f = signal.filtfilt(b, a, data)
    if plot:
        plot_freq_response(b, a, fs)

    return f


#######################################
def thz_pulse(t, tau=0.15, loc=0):
    return t * norm.pdf(t, scale=tau)


def thz_pulse2(t, tau=0.15):
    return (t / tau ** 2) * (np.exp(-t ** 2 / tau ** 2))


# most similar to measurement values (?)
tau = 0.28  # mm/ps
dt = 0.05  # Ps # 0.28 ps ~ 1.80 THz, sampling rate

# tau = 0.1  # mm/ps
# dt = 0.01  # Ps

c0 = c * 10**3 / 10**12

t = np.arange(0, 1000, dt)
t_r = 46.45
t_0 = t_r + 0

n = [1.00, 1.50, 2.80, 1.50]
d = [0.040, 0.640, 0.075]
#d = np.array([0.150, 0.100, 0.200])

t_1 = 2 * (d[0] * n[1]) / c0 + t_0
ab1 = 1  # np.exp(-0.5)
t_2 = 2 * (d[1] * n[2]) / c0 + t_1
ab2 = 1  # np.exp(-0.5)
t_3 = 2 * (d[2] * n[3]) / c0 + t_2
ab3 = 1  # np.exp(-0.5)

print("1-layer:", t_1 - t_0, "2-layer:", t_2 - t_1, "3-layer:", t_3 - t_2)
r0 = (n[0] - n[1]) / (n[0] + n[1])
t01 = 2 * n[0] / (n[0] + n[1])
t10 = 2 * n[1] / (n[0] + n[1])
r1 = (n[1] - n[2]) / (n[1] + n[2])
t12 = 2 * n[1] / (n[1] + n[2])
t21 = 2 * n[2] / (n[1] + n[2])
r2 = (n[2] - n[3]) / (n[2] + n[3])
t23 = 2 * n[2] / (n[2] + n[3])
t32 = 2 * n[3] / (n[2] + n[3])

r3 = (n[3] - n[0]) / (n[3] + n[0])
r4 = -1

y = r4 * thz_pulse2(t - t_r, tau)  # Referenz
y2 = r0 * thz_pulse2(t - t_0, tau) + \
     ab1 * t01 * r1 * t10 * thz_pulse2(t - t_1, tau) + \
     ab1 * ab2 * t01 * t12 * r2 * t10 * t21 * thz_pulse2(t - t_2, tau) + \
     ab1 * ab2 * ab3 * t01 * t12 * t23 * r3 * t32 * t21 * t10 * thz_pulse2(t - t_3, tau)

# Noise
# y  += np.random.random(len(t)) * np.max(np.abs(y))
#y2 += np.random.random(len(t)) * np.max(np.abs(y2)) * 0.002
#y += np.random.random(len(t)) * np.max(np.abs(y)) * 0.002

# y = np.concatenate((y, np.zeros(5*len(y))))
# y2 = np.concatenate((y2, np.zeros(5*len(y2))))


Y = np.fft.fft(y)
Y2 = np.fft.fft(y2)
freqs = np.fft.fftfreq(len(t), dt)
idx = (freqs >= 0.00) * (freqs <= 1.75)

print(freqs[idx][10]-freqs[idx][11])
print(freqs[idx][0], freqs[idx][1], freqs[idx][-1])
print(len(freqs[idx]))
simple_p = -2 * pi * ((n[1] - 1) * d[0] + (n[2] - 1) * d[1] + (n[3] - 1) * d[2]) * freqs[idx] / c0

# np.log10 = lambda x: x / 20
plt.figure("Amplitude frequency domain (Sim)")
plt.plot(freqs[idx], 20 * np.log10(np.abs(Y[idx])), label="Reference")
plt.plot(freqs[idx], 20 * np.log10(np.abs(Y2[idx])), label="Sample")
plt.plot(freqs[idx], 20 * np.log10(np.abs(Y2[idx] / Y[idx])), label="Sam/Ref")
plt.xlabel("Frequency (THz)")
plt.ylabel("Amplitude (dB)")
plt.legend()

plt.figure("Raw phase frequency domain (Sim)")
p_sam = np.arctan2(Y2[idx].imag, Y2[idx].real)
p_ref = np.arctan2(Y[idx].imag, Y[idx].real)
#p_uwrap_sam = np.angle(Y2[idx])
#p_uwrap_ref = np.angle(Y[idx])
p_diff = p_sam - p_ref
#p_unwrap_diff = fix_discontinuities(p_unwrap_diff)
print(p_diff[1000])
print(freqs[idx][1000])
plt.plot(freqs[idx], p_ref, label="Reference")
plt.plot(freqs[idx], p_sam, label="Sample")
plt.plot(freqs[idx], p_diff, label="Sam - Ref")
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (rad)")
plt.legend()


def shift(p_uwrap):
    slice_ = (40, 60)
    a = np.mean(np.diff(p_uwrap)[slice_[0]:slice_[1]])
    b = p_uwrap[slice_[0]] - a * slice_[0]
    p_uwrap -= b

    return p_uwrap


# np.unwrap = lambda x: x
plt.figure("Unwrapped phase frequency domain (Sim)")
p_unwrap_sam = shift(np.unwrap(np.angle(Y2[idx])))
p_unwrap_ref = shift(np.unwrap(np.angle(Y[idx])))
p_unwrap_diff = p_unwrap_sam - p_unwrap_ref
plt.plot(freqs[idx], p_unwrap_ref, label="Reference")
plt.plot(freqs[idx], p_unwrap_sam, label="Sample")
plt.plot(freqs[idx], p_unwrap_diff, label="Sam - Ref")
plt.plot(freqs[idx], simple_p, label="Simple phase model")
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (rad)")
plt.legend()

zero_pad = 0
Y = np.concatenate((Y, np.zeros(zero_pad)))
Y2 = np.concatenate((Y2, np.zeros(zero_pad)))

phase_y, phase_y2 = np.angle(Y), np.angle(Y2)
# phase_y2[freqs > 1.5] = 0
# phase_y[freqs > 1.5] = 0
Y = np.abs(Y) * np.exp(1j * phase_y)
Y2 = np.abs(Y2) * np.exp(1j * phase_y2)

y = np.fft.ifft(Y)
y2 = np.fft.ifft(Y2)
t_fun = np.fft.ifft(Y2 / Y)
t_fun = np.roll(t_fun, 1000)

fs = 1 / dt

lc, hc = 0.22, 1.72

# y2 = butter_lowpass(y2, hc, fs, order=1, plot=True)
# y2 = butter_highpass(y2, lc, fs, order=7, plot=True)
# y2 = butter_bandpass(y2, lc, hc, fs, order=3, plot=True)

plt.figure("Amplitude time domain (Sim)")
plt.plot(t, y, label="Reference")
plt.plot(t, y2, label="Sample")
plt.plot(t, t_fun, label="transfer func")
# plt.plot(t, t_fun_filt, label="transfer func filtered")
plt.legend()
plt.show()

dx = 1
dy = 1
tilt = 8 / 180 * np.pi

k = 0
for i in range(1):
    for j in range(1):
        k += 1
        t_2 = 2 * (d[1] * n[2]) / c0 + t_1 + np.tan(tilt) * i * dx / c0
        y2 = r0 * thz_pulse2(t - t_0, tau) + \
             ab1 * t01 * r1 * t10 * thz_pulse2(t - t_1, tau) + \
             ab1 * ab2 * t01 * t12 * r2 * t10 * t21 * thz_pulse2(t - t_2, tau) + \
             ab1 * ab2 * ab3 * t01 * t12 * t23 * r3 * t32 * t21 * t10 * thz_pulse2(t - t_3, tau)
        y2 += np.random.random(len(t)) * np.max(np.abs(y2)) * 0.02
        temp = np.vstack((t, y2)).T
