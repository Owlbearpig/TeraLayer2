import matplotlib.pyplot as plt
from functions import get_full_measurement
from numpy.fft import ifft, fftfreq, fft
from numpy import zeros
from scipy import signal
import numpy as np
from scipy.signal import windows

GHz = 10 ** 9
THz = 10 ** 12
fs = 2 * 2.4 * THz
fq = fs / 2

def mean_filter(x):
    k = 10
    kern = np.ones(2 * k + 1) / (2 * k + 1)
    x = np.convolve(x, kern, mode='same')
    return x

def prepare_ref():
    f, r, b, s = get_full_measurement(sample_file_idx=56, f_slice=(400, 1600))

    """
    f_slice = (f > -400*GHz)*(f < 3100*GHz)
    f = f[f_slice]
    ref = (r[f_slice] - b[f_slice])
    """

    x = r# - b

    plt.figure()
    plt.plot(f, 20 * np.log10(np.abs(x)), label="ref")
    plt.legend()
    plt.show()

    zero_pad = 20000
    x = np.concatenate((x, zeros(zero_pad)))

    shift = 2000
    x = np.roll(x, shift)

    window = windows.tukey(int(len(r)), 0.01)
    window = np.concatenate((zeros(shift), window, zeros(zero_pad-shift)))

    plt.figure()
    plt.plot(window, label="window")
    plt.plot(x * window, label="signal*window")
    plt.plot(x, label="signal")
    plt.xlabel("frequency")
    plt.legend()
    plt.show()

    x = x * window

    """
    plt.figure()
    plt.plot(fft(t_fun_ifft), label="t_fun_fft")
    plt.legend()
    plt.show()
    """

    x = ifft(x)
    x = np.roll(x, shift)

    #x = highpassfilt(x)
    #x = lowpassfilt(x)
    #x /= max(abs(x))
    t = np.linspace(0, len(x) / fs, len(x)) * 10 ** 12
    plt.figure()
    plt.plot(t, x, label="with filter")
    plt.xlabel("time")
    plt.legend()
    plt.show()

    exit()


def plot_freq_response(b, a, fs=fs, worN=8000):
    # Plot the frequency response.
    w, h = signal.freqz(b, a, worN=worN)
    plt.figure()
    plt.plot(0.5 * fs * w / (THz * np.pi), 20 * np.log10(np.abs(h)), 'b')
    # plt.plot(highcut, 0.5*np.sqrt(2), 'ko')
    # plt.axvline(highcut, color='k')
    # plt.xlim(0, 0.5*fs / GHz)
    plt.title("Filter Frequency Response")
    plt.xlabel('Frequency (THz)')
    plt.grid()
    plt.show()


def highpassfilt(y):
    # fc = 0.22*THz
    fc = 0.1 * THz
    w = fc / (fs / 2)
    b, a = signal.butter(1, w, btype='high')
    y = signal.filtfilt(b, a, y)
    plot_freq_response(b, a)
    return y


def lowpassfilt(y):
    # fc = 1.72 * THz
    fc = 1.5 * THz
    w = fc / (fs / 2)
    b, a = signal.butter(4, w, btype='low')
    y = signal.filtfilt(b, a, y)
    plot_freq_response(b, a)
    return y

prepare_ref()

f, r, b, s = get_full_measurement(sample_file_idx=56)

pos_freqs = f > 0
neg_freqs = f < 0
print("neg. freq. cnt", sum(neg_freqs))
print("pos. freq. cnt:", sum(pos_freqs))

t_fun = (s - b) / (r - b)

plt.figure()
plt.plot(f, 20 * np.log10(np.abs(r)), label="ref")
plt.plot(f, 20 * np.log10(np.abs(s)), label="sam")
plt.plot(f, 20 * np.log10(np.abs(b)), label="bg")
plt.plot(f, 20*np.log10(np.abs(t_fun)), label="t_func")
plt.legend()
plt.show()

# f = fftfreq()
zero_pad = zeros(29367)
neg_freq_zero_pad = zeros(sum(pos_freqs) - sum(neg_freqs))
x = np.concatenate((t_fun[f == 0], t_fun[pos_freqs], t_fun[neg_freqs], neg_freq_zero_pad, zero_pad))

shift = 1000
x = np.roll(x, shift)

window = windows.tukey(int(len(t_fun)*1), 0.7)
window = np.concatenate((zeros(shift), window))
window = np.concatenate((window, zeros(len(x)-len(window))))
plt.plot(window, label="window")
plt.plot(x*window, label="signal*window")
plt.plot(x, label="signal")
plt.legend()
plt.show()

x = x*window

#x = np.concatenate((zero_pad, t_fun[436:]))

"""
plt.figure()
plt.plot(fft(t_fun_ifft), label="t_fun_fft")
plt.legend()
plt.show()
"""

t_fun_ifft = ifft(x)
t_fun_nofilt = t_fun_ifft.copy()
x = np.roll(t_fun_ifft, 1000)

plt.figure()

#t_fun_ifft = highpassfilt(t_fun_ifft)
#x = lowpassfilt(x)
x /= max(abs(x))

t = np.linspace(0, len(x) / fs, len(x))
#plt.plot(t_fun_nofilt, label="no filter")
plt.plot(t * 10**12, x, label="with filter")
plt.legend()
plt.show()
