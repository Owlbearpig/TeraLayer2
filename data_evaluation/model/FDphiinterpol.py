import numpy as np
from numpy import zeros
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import windows
import os

if os.name == 'posix':
    ref_file = r"/home/alex/PycharmProjects/TeraLayer2/data_evaluation/matlab_enrique/Data/ref_1000x.csv"
    bkg_file = r"/home/alex/PycharmProjects/TeraLayer2/data_evaluation/matlab_enrique/Data/BG_1000.csv"
else:
    ref_file = r"E:\Projects\TeraLayer2\data_evaluation\matlab_enrique\Data\ref_1000x.csv"
    bkg_file = r"E:\Projects\TeraLayer2\data_evaluation\matlab_enrique\Data\BG_1000.csv"


def plot_freq_response(b, a, fs, worN=8000):
    # Plot the frequency response.
    w, h = signal.freqz(b, a, worN=worN)
    plt.figure()
    plt.plot(0.5 * fs * w / (10 ** 12 * np.pi), 20 * np.log10(np.abs(h)), 'b')
    # plt.plot(highcut, 0.5*np.sqrt(2), 'ko')
    # plt.axvline(highcut, color='k')
    # plt.xlim(0, 0.5*fs / GHz)
    plt.title("Filter Frequency Response")
    plt.xlabel('Frequency (THz)')
    plt.grid()
    plt.show()


def butter_highpass(data, lowcut, fs, order=7, plot=False):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='high')

    f = signal.filtfilt(b, a, data)
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


def do_fft(t, y):
    n = len(y)
    dt = np.float(np.mean(np.diff(t)))
    Y = np.fft.fft(y, n)
    f = np.fft.fftfreq(len(t), dt)
    idx_range = f > 0

    return f[idx_range], Y[idx_range]


def do_ifft(Y):
    y = np.fft.ifft(Y)

    return y


def freq_axis():
    df = pd.read_csv(ref_file)

    return df.values[:, 0] * 10 ** 6


def e_field(file_path, sub_bkg=False, phi_interp=True):
    df = pd.read_csv(file_path)

    if sub_bkg:
        df_bkg = pd.read_csv(bkg_file)

        r = df.values[:, 1] - df_bkg.values[:, 1]
        phi = df.values[:, 2] - df_bkg.values[:, 2]
    else:
        r = df.values[:, 1]
        phi = df.values[:, 2]

        if phi_interp:
            x = np.arange(700, 820)
            (a, b) = np.polyfit(x, np.unwrap(phi)[700:820], 1)

            x = np.arange(0, len(phi))
            """
            plt.plot(np.unwrap(phi))
            plt.plot(x*a+b)
            plt.show()
            """
            phi = x * a + b

    return r * np.exp(1j * phi)


freqs = freq_axis()

freq_slice = (0.00 * 10 ** 12 < freqs) * (freqs < 4.50 * 10 ** 12)

R_ref = e_field(ref_file, sub_bkg=False)
R_bkg = e_field(bkg_file, phi_interp=False)
R_ref_nobkg = e_field(ref_file, sub_bkg=True)

plt.plot(freqs[freq_slice], 20 * np.log10(np.abs(R_ref))[freq_slice], label="ref")
plt.plot(freqs, 20 * np.log10(np.abs(R_bkg)), label="R_bkg")
# plt.plot(freqs, np.log10(np.abs(R_ref_nobkg)), label="ref - bkg")
plt.legend()
plt.show()

R_ref = R_ref[freq_slice]

f_max = freqs[freq_slice].max()
fs = 2 * f_max


zero_pad = len(R_ref) * 10
y = np.concatenate((R_ref[:len(R_ref)//2], zeros(zero_pad), R_ref[len(R_ref)//2:-1]))

sl = len(y)
fs = np.mean(np.diff(freqs)) * sl

y = np.fft.ifft(y)

lc = 0.20
hc = 3.00

y_filt = butter_highpass(y, order=3, lowcut=lc * 10 ** 12, plot=True, fs=fs)
y_filt = y
# y_filt = butter_bandpass(y, order=3, lowcut=lc*10**12, highcut=hc*10**12, plot=True, fs=fs)

dt = 1 / fs
t = np.linspace(0, sl * dt, sl)

plt.plot(y, label="ref")
plt.plot(y_filt, label="ref filt")
plt.legend()
plt.show()

Y = np.fft.fft(y)
Y_filt = np.fft.fft(y_filt)

#plt.plot(f, 20 * np.log10(np.abs(Y)), label="ref freq domain")
plt.plot(20 * np.log10(np.abs(Y_filt)), label="ref hp filt freq domain")
plt.legend()
plt.show()
