import matplotlib.pyplot as plt
from functions import get_full_measurement
from numpy.fft import ifft, fftfreq, fft, rfft, irfft
from numpy import zeros
from scipy import signal
import numpy as np
from scipy.signal import windows
from consts import c0, um

GHz = 10 ** 9
THz = 10 ** 12
#fs = 2 * 2.4 * THz

def do_fft(t, y):
    n = len(y)
    dt = np.float(np.mean(np.diff(t)))
    Y = np.fft.fft(y, n)
    f = np.fft.fftfreq(len(t), dt)
    idx_range = f > 0

    return f[idx_range], Y[idx_range]


def mean_filter(x, k=15):
    kern = np.ones(2 * k + 1) / (2 * k + 1)
    x = np.convolve(x, kern, mode='same')
    return x



def prepare_ref():
    thz_path = r"E:\measurementdata\BraggMirror\msr_220531_BraggGitterOnSub\THz\Ref3x3_daten_linescan0.1mm_3xnebeneinander0.4mm_10Avg_constNormal\2022-05-31T17-13-07.118202-Ref3x3-[241]-[9.4,-5.95,1.05]-[1.0,0.0,0.0,0.0]-delta[0.011mm-0.0deg]-avg10.txt"
    ref_data = np.loadtxt(thz_path)
    #f, r, b, s = get_full_measurement(sample_file_idx=56, f_slice=(420, 1375))
    f, r, b, s = get_full_measurement(sample_file_idx=56, f_slice=(600, 1050))
    f_ref, Y_ref = do_fft(ref_data[:, 0], ref_data[:, 1])
    #f_idx = (f_ref < 1.05)*(f_ref > 0.6)
    #f_ref, Y_ref = f_ref[f_idx], np.concatenate((Y_ref[f_idx], zeros(8000)))
    """
    f_slice = (f > -400*GHz)*(f < 3100*GHz)
    f = f[f_slice]
    ref = (r[f_slice] - b[f_slice])
    """

    x = r - b

    #x[1300] *= 100
    t_fun = (s - b) / (r - b)

    plt.figure()
    #plt.plot(20 * np.log10(np.abs(Y_ref)), label="thz ref")
    plt.plot(f/GHz, 20 * np.log10(np.abs(x)), label="ref")
    plt.plot(f/GHz, 20 * np.log10(np.abs(t_fun)), label="t_fun")
    plt.plot(f/GHz, 20 * np.log10(np.abs(b)), label="bkgrnd")
    #plt.xlim((-1000, 5000))
    plt.legend()
    plt.show()

    zero_pad = 50000
    x = np.repeat(x, 20)
    x = np.concatenate((x, zeros(zero_pad)))
    shift = 0#600
    x = np.roll(x, shift)

    plt.figure()
    plt.plot(np.abs(x), label="signal")

    window_en = True
    if window_en:
        #window = windows.tukey(int(len(r)), 0.2)
        window = windows.tukey(int(len(r)*20), 0.7)
        #window = np.random.randint(2, size=len(window))
        #window = windows.exponential(int(len(r)), tau=(len(x)/2)/(60/8.69))
        f_offset = 0
        window = np.concatenate((zeros(shift+f_offset), window))
        window = np.concatenate((window, zeros(zero_pad-shift-f_offset)))
        x *= window

        plt.plot(window, label="window")
        plt.plot(np.abs(x * window), label="signal*window")

    mean_filt_en = False
    if mean_filt_en:
        k = 1
        x = mean_filter(x, k=k)
        plt.plot(np.abs(x), label=f"mean filtered k={k}")

    savgol_filt_en = False
    if savgol_filt_en:
        from scipy.signal import savgol_filter
        x = savgol_filter(x, 91, 3)  # window size 51, polynomial order 3
        plt.plot(np.abs(x), label=f"savgol_filter filtered")

    plt.xlabel("frequency")
    plt.legend()
    plt.show()



    """
    plt.figure()
    plt.plot(fft(t_fun_ifft), label="t_fun_fft")
    plt.legend()
    plt.show()
    """

    x = ifft(x)
    #x = ifft(x)
    x = np.roll(x, 10000)

    #x = highpassfilt(x)
    #x = lowpassfilt(x)

    #x = highpassfilt(x)
    #x = lowpassfilt(x)
    x /= max(abs(x))
    t = np.linspace(0, len(x) / fs, len(x)) * 10 ** 12

    plt.figure()
    plt.plot(t[:32568], x[:32568], label="with filter")
    plt.xlabel("time")
    plt.legend()
    plt.show()

    return x[:32568]


def plot_freq_response(b, a, fs, worN=8000):
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
    fc = 0.32 * THz
    w = fc / (fs / 2)
    b, a = signal.butter(2, w, btype='high')
    y = signal.filtfilt(b, a, y)
    plot_resp_en = False
    if plot_resp_en:
        plot_freq_response(b, a)
    return y


def lowpassfilt(y):
    # fc = 1.72 * THz
    fc = 1.3 * THz
    w = fc / (fs / 2)
    b, a = signal.butter(4, w, btype='low')
    y = signal.filtfilt(b, a, y)
    plot_resp_en = False
    if plot_resp_en:
        plot_freq_response(b, a)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4, plot = False):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    f = signal.filtfilt(b, a, data)
    if plot:
        plot_freq_response(b,a, fs)

    return f

#x_ref = prepare_ref()
f, r, b, s = get_full_measurement(sample_file_idx=0, f_slice=(-100, 2100))

r, b, s = np.zeros_like(r), np.zeros_like(b), np.zeros_like(s)
for i in range(101):
    _, ri, bi, si = get_full_measurement(sample_file_idx=i, f_slice=(-100, 2100))
    r += ri
    b += bi
    s += si

r /= 100
b /= 100
s /= 100

print(np.mean(np.diff(f))/GHz)
z_pad = len(r)*15
sl = z_pad / 2
fs = 2 * (np.mean(np.diff(f))/GHz) * sl * THz / 1000

pos_freqs = f > 0
neg_freqs = f < 0
print("neg. freq. cnt", sum(neg_freqs))
print("pos. freq. cnt:", sum(pos_freqs))

t_fun = (s - b) / (r - b)

x = t_fun
plt.figure()
plt.plot(f, 20 * np.log10(np.abs(r)), label="ref")
plt.plot(f, 20 * np.log10(np.abs(s)), label="sam")
plt.plot(f, 20 * np.log10(np.abs(b)), label="bg")
plt.plot(f, 20 * np.log10(np.abs(t_fun)), label="t_func")
plt.legend()
plt.show()

zero_pad = zeros(z_pad)
x = np.concatenate((x, zero_pad))

# f = fftfreq()

neg_freq_zero_pad = zeros(sum(pos_freqs) - sum(neg_freqs))
#x = np.concatenate((t_fun[f == 0], t_fun[pos_freqs], t_fun[neg_freqs], neg_freq_zero_pad, zero_pad))

#shift = 1000
#x = np.roll(x, shift)

window = windows.tukey(int(len(t_fun)*1), 0.4)
#window = np.concatenate((zeros(shift), window))
window = np.concatenate((window, zeros(len(x)-len(window))))

plt.figure()
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

x = ifft(x)
x = np.roll(x, 1000)

lc, hc = 0.1, 2.0

x = butter_bandpass_filter(x, lowcut=lc*THz, highcut=hc*THz, plot=True, fs=fs)
#x = highpassfilt(x)
#x = lowpassfilt(x)

#x /= max(abs(x))

t = np.linspace(0, len(x) / fs, len(x))
t *= 10**12

def obj_function(p, y_data):
    y_mod = np.zeros_like(y_data.real)

    y_mod[int(p[0]*len(y_data))] = 1
    y_mod[int(p[1]*len(y_data))] = -1
    y_mod[int(p[2]*len(y_data))] = -0.95
    y_mod[int(p[3]*len(y_data))] = -0.85

    y_mod = butter_bandpass_filter(y_mod, lowcut=0.3 * THz, highcut=1.5 * THz, plot=False, fs=fs)

    return np.sum(y_mod**2*(y_mod-y_data.real)**2)


"""
from scipy.optimize import minimize
x0 = np.array([1017/len(t), 1034/len(t), 1294/len(t), 1311/len(t)])
x0 = 0.5*np.ones_like(x0)
res = minimize(obj_function, x0=x0, args=(x))

p4 = np.linspace(0, 0.1, len(x)-2)
plt.figure()
f = []
for p4_ in p4:
    p = np.array([1017/len(t), 1034/len(t), 1294/len(t), p4_])
    f.append(obj_function(p, x))
plt.plot(np.round(p4*len(x)), f)
plt.show()

print(res)
"""
"""
x_mod = np.zeros_like(x)
x_mod[1019] = 1
x_mod[1032] = -1
x_mod[1274] = -1
x_mod[1289] = -1
"""
"""
# sample #10
x_mod = np.zeros_like(x)
x_mod[1028] = 1
x_mod[1047] = -1
x_mod[1398] = -1
x_mod[1420] = -1
"""
# sample #80
x_mod = np.zeros_like(x)
#x_mod[1040] = -0.04
x_mod[1028] = 0.080
x_mod[1057] = -0.075
x_mod[1488] = -0.057
x_mod[1511] = -0.060

x_mod = butter_bandpass_filter(x_mod, lowcut=lc*THz, highcut=hc*THz, plot=False, fs=fs)

#diff = np.diff([t[1021], t[1030], t[1289], t[1304]])
diff = np.diff([t[1028], t[1057], t[1488], t[1511]])
dt = np.mean(np.diff(t))
print(f"delta t: {dt}")
print(f"t diff (ps): {diff}")
n = np.array([1.5, 2.8, 1.5])
angle = 8*np.pi/180

print(f"thicknesses (um): {0.5*(diff/10**12)*(c0/n) * um * np.cos(angle)}")

plt.figure()
plt.plot(x, label="t fun")

#x_mod /= max(abs(x_mod))

plt.plot(x_mod, label="x mod")
plt.xlim((500, 2000))
plt.legend()
plt.show()


