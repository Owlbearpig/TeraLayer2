from consts import *
from numpy import cos, sin, exp, array, arcsin, pi, conj, sum
import matplotlib.pyplot as plt


class ReflectanceModel:
    def __init__(self, freq_axis):
        self.freqs = freq_axis
        self.lam = c0 / freq_axis
        self.s_consts = self.set_semi_consts()

    def set_semi_consts(self):
        """
        some values only need to be calculated once, they are indep. of p
        :return: None
        """
        self.the = array([thea, 0, 0, 0, 0])
        for i in range(0, 4):
            self.the[i + 1] = arcsin(n[i] * sin(self.the[i]) / n[i + 1])

        return *list(map(self._a, range(4))), \
               *list(map(self._b, range(4))), \
               *list(map(self._f, range(0, 3)))

    def _a(self, k):
        enumerator = n[k] * cos(self.the[k + 1]) - n[k + 1] * cos(self.the[k])
        denum = n[k + 1] * cos(self.the[k]) + n[k] * cos(self.the[k + 1])

        return enumerator / denum

    def _b(self, k):
        enumerator = n[k + 1] * cos(self.the[k]) - n[k] * cos(self.the[k + 1])
        denum = n[k] * cos(self.the[k + 1]) + n[k + 1] * cos(self.the[k])

        return enumerator / denum


    def _f(self, k):
        """
        calculate unsigned exponents
        :param k: di index
        :return: array with length == len(lam)
        """

        return 1j * 2 * pi * n[k + 1] / self.lam


    def _calculation(self, p, s_consts):
        """
        Note: g(k) = c(k) * d(k) - a(k) * b(k) in paper calculation is always 1.
        :param p: parameters
        :param s_consts: values which only need to be evaluated once. (Independent of p)
        :return: wavelength resolved reflectance R
        """
        a0, a1, a2, a3, b0, b1, b2, b3, f0_0, f0_1, f0_2 = s_consts
        f0, f1, f2 = p[0] * f0_0, p[1] * f0_1, p[2] * f0_2

        # the 8 terms of M_12
        t0_12 = b0 * exp(-f2 - f1 - f0)
        t1_12 = -a2 * b3 * b0 * exp(f2 - f1 - f0)
        t2_12 = b1 * exp(-f2 - f1 + f0)
        t3_12 = -a2 * b3 * b1 * exp(f2 - f1 + f0)
        t4_12 = -a1 * b0 * b2 * exp(-f0 - f2 + f1)
        t5_12 = b2 * exp(-f2 + f1 + f0)
        t6_12 = -a1 * b0 * b3 * exp(-f0 + f2 + f1)
        t7_12 = b3 * exp(f2 + f1 + f0)

        # the 8 terms of M_22
        t0_22 = -a3 * b0 * exp(-f1 - f0 - f2)
        t1_22 = -b1 * a3 * exp(-f1 + f0 - f2)
        t2_22 = -a2 * b0 * exp(f2 - f0 - f1)
        t3_22 = -a2 * b1 * exp(f2 - f1 + f0)
        t4_22 = a1 * a3 * b2 * b0 * exp(-f0 - f2 + f1)
        t5_22 = -a3 * b2 * exp(-f2 + f0 + f1)
        t6_22 = exp(f2 + f1 + f0)  # weird term
        t7_22 = -a1 * b0 * exp(f2 + f1 - f0)

        m_12 = t0_12 + t1_12 + t2_12 + t3_12 + t4_12 + t5_12 + t6_12 + t7_12
        m_22 = t0_22 + t1_22 + t2_22 + t3_22 + t4_22 + t5_22 + t6_22 + t7_22

        r = m_12 / m_22

        return r

    def calc_r(self, p):
        s_consts = self.s_consts
        return self._calculation(p, s_consts)

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    print(low, high)
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


if __name__ == '__main__':
    from scipy.signal import windows, freqz

    x_axis = np.arange(0, 1500, 1).astype(np.float64)
    freqs = x_axis * GHz

    new_r_model = ReflectanceModel(freq_axis=freqs)
    p0 = array([2860.0, 4997.0, 0.0]) * um_to_m
    p0 = array([0.0, 5000.0, 0.0]) * um_to_m
    p1 = array([0.0, 2500.0, 0.0]) * um_to_m
    p0 = array([2860.0, 4997.0, 0.0]) * um_to_m
    r_model0 = new_r_model.calc_r(p0)
    r_model1 = new_r_model.calc_r(p1)

    enable_window = False
    if enable_window:
        for r_mod in [r_model0, r_model1]:
            window = windows.tukey(len(r_mod), 0.01, sym=False)
            r_mod *= window

    p0_str, p1_str = round(p0*um), round(p1*um)

    df = np.mean(np.diff(freqs))
    dt = 1 / df

    t = x_axis * dt

    enable_filter = False
    if enable_filter:
        fs = df
        lowcut = 10**6
        highcut = 0.45 * GHz

        b, a = butter_lowpass(cutoff=lowcut, fs=fs, order=9)
        w, h = freqz(b, a, worN=len(freqs))

        r_model0 *= h

    plot_phase = False
    if plot_phase:
        plt.figure()
        plt.plot(freqs / GHz, np.unwrap(np.angle(r_model0)), label=f"r_model0 angle(r) {p0_str}")
        #plt.plot(freqs / GHz, np.unwrap(np.angle(r_model1)), label=f"r_model0 angle(r) {p1_str}")
        plt.legend()
        plt.xlabel("frequency (GHz)")

    plt.figure()
    plt.plot(freqs / GHz, np.log10(np.abs(r_model0)), label=f"model np.log10(abs(r_model0)) {p0_str}")
    #plt.plot(freqs / GHz, np.log10(np.abs(r_model1)), label=f"model np.log10(abs(r_model1)) {p1_str}")
    plt.legend()
    plt.xlabel("frequency (GHz)")



    r_model_fd0 = np.fft.fft(r_model0, len(r_model0))
    r_model_fd1 = np.fft.fft(r_model1, len(r_model1))

    plt.figure()
    plt.plot(t, (np.abs(r_model_fd0)), label=f"r_model_fd0 {p0_str}")
    #plt.plot(t * c0, (np.abs(r_model_fd1)), label=f"r_model_fd1 {p1_str}")
    plt.legend()
    plt.xlabel("time (s)")
    plt.show()


