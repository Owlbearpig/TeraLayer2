import numpy as np
from numpy import zeros
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
from scipy.signal import windows
import os

THz = 10 ** 12

if os.name == 'posix':
    bkg_file = Path(r"/home/alex/PycharmProjects/TeraLayer2/data_evaluation/matlab_enrique/Data/BG_1000.csv")
    ref_file = Path(r"/home/alex/PycharmProjects/TeraLayer2/data_evaluation/matlab_enrique/Data/ref_1000x.csv")
    sam_file = Path(r"")
else:
    bkg_file = Path(r"E:\Projects\TeraLayer2\data_evaluation\matlab_enrique\Data\BG_1000.csv")
    ref_file = Path(r"E:\Projects\TeraLayer2\data_evaluation\matlab_enrique\Data\ref_1000x.csv")
    sam_file = Path(r"E:\Projects\TeraLayer2\data_evaluation\matlab_enrique\Data\Kopf_1x\Kopf_1x_0009")


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

def butter_lowpass(data, highcut, fs, order=7, plot=False):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = signal.butter(order, high, btype='low')

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


def freq_axis(freq_range=None):
    df = pd.read_csv(ref_file)
    freqs = df.values[:, 0] * 10 ** 6

    if freq_range is not None:
        freq_slice = (freq_range[0] * THz <= freqs) * (freqs <= freq_range[1] * THz)
        return freqs[freq_slice]
    else:
        return freqs


def e_field(file, sub_bkg=False, phi_interp=True, freq_range=None):
    df = pd.read_csv(file)

    freqs = df.values[:, 0] * 10 ** 6

    if sub_bkg:
        df_bkg = pd.read_csv(bkg_file)

        r = df.values[:, 1] - df_bkg.values[:, 1]
        phi = df.values[:, 2] - df_bkg.values[:, 2]
    else:
        r = df.values[:, 1]
        phi = df.values[:, 2]

        if phi_interp:
            fit_range = (0.460 * THz, 0.595 * THz)
            fit_slice = (fit_range[0] <= freqs) * (freqs <= fit_range[1])

            (a, b) = np.polyfit(freqs[fit_slice], np.unwrap(phi)[fit_slice], 1)

            phi_lin = freqs * a  # + b # add phase offset for "other" pulse.

            plt.plot(freqs, np.unwrap(phi), label=f"unwrapped phase {file.stem}")
            #plt.plot(freqs[fit_slice], np.unwrap(phi)[fit_slice], label=f"unwrapped phase (fitted part) {file.stem}")
            if not "BG" in file.stem:
                plt.plot(freqs, phi_lin, label=f"a*x {file.stem}")
                plt.plot(freqs, phi_lin + b, "-.", label=f"a*x + b {file.stem}")
            plt.vlines(fit_range, min(phi_lin), max(phi_lin), ls='--')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("UnwrappedPhase")

            phi = phi_lin

    if freq_range is not None:
        freq_slice = (freq_range[0] * THz <= freqs) * (freqs <= freq_range[1] * THz)
        field = (r * np.exp(1j * phi))[freq_slice]
    else:
        field = r * np.exp(1j * phi)

    return field


def preprocess(file, ret_freqdomain=True, phi_interp=True, freq_range=None):


    freqs = freq_axis(freq_range=freq_range)
    y = e_field(file, sub_bkg=False, freq_range=freq_range, phi_interp=phi_interp)

    freq_rez = np.mean(np.diff(freqs))

    # zero pad at end k * len(y)
    k = 3
    pad_len = len(y)*k
    y = np.concatenate((y, zeros(pad_len)))
    freqs = np.concatenate((freqs, np.arange(freqs.max(), freqs.max()+freq_rez*pad_len, freq_rez)))

    # add zeroes down to dc (if necessary)
    #freqs = np.concatenate((np.arange(0, freqs.min(), freq_rez), freqs))
    #y = np.concatenate((zeros(len(freqs)-len(y)), y))

    # conjugate transform
    en_conj_tform = False
    if en_conj_tform:
        y = np.concatenate((np.flip(np.conjugate(y)), y))
        freqs = np.concatenate((-np.flip(freqs), freqs))

    if ret_freqdomain:
        return freqs, y
    else:
        lc, hc = 0.35, 2.50
        fs = 2 * freqs.max()
        sl = len(y)

        y = np.fft.ifft(y)
        y = butter_highpass(y, order=2, lowcut=lc * 10 ** 12, plot=True, fs=fs)

        dt = 1 / fs
        t = np.linspace(0, sl * dt, sl)

        plt.plot(t * 10 ** 12, y, label="signal time domain")
        # plt.plot(y, label="ref")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (a. u.)")
        plt.legend()
        plt.show()

        return t, y


plt.figure("Phase plot")

freq_range = (0.22, 2.15) # 0.22
freqs, y_ref = preprocess(ref_file, phi_interp=True, freq_range=freq_range)
_, y_sam = preprocess(sam_file, phi_interp=True, freq_range=freq_range)
_, _ = preprocess(bkg_file, phi_interp=True, freq_range=freq_range) # just for plotting

plt.legend()
plt.show()

y = y_sam / y_ref
y = np.nan_to_num(y)

plt.figure("Frequency domain")
plt.plot(freqs, 20 * np.log10(np.abs(y_ref)), label="ref")
plt.plot(freqs, 20 * np.log10(np.abs(y_sam)), label="sam")
plt.plot(freqs, 20 * np.log10(np.abs(y)), label="t fun")
plt.ylabel("Amplitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.legend()
plt.show()

lc, hc = 0.22, 1.9
sl = len(y)
fs = 2 * freq_range[1] * THz

y = np.fft.ifft(y)
#y = butter_highpass(y, order=2, lowcut=lc * 10 ** 12, plot=True, fs=fs)
#y = butter_lowpass(y, order=7, highcut=hc * 10 ** 12, plot=True, fs=fs)
#y = butter_bandpass(y, lc * 10 ** 12, hc * 10 ** 12, fs, order=7, plot=True)

dt = 1 / fs
t = np.linspace(0, sl * dt, sl)

y = np.roll(y, 1000)

plt.figure("Time domain")
plt.plot(t * 10 ** 12, y, label="t fun time domain")
plt.xlabel("Time (ps)")
plt.ylabel("Amplitude (a. u.)")
plt.legend()
plt.show()
