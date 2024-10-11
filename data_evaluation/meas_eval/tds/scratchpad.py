import numpy as np
import matplotlib.pyplot as plt

from functions import filtering
from consts import *
import pandas as pd
from model.tmm_package import tmm_package_wrapper
from model.refractive_index import get_n

from mpl_settings import *
from numpy import nan_to_num
from numpy.fft import ifft

def do_fft(data_td):
    t, y = data_td[:, 0], data_td[:, 1]
    n = len(y)
    dt = float(np.mean(np.diff(t)))
    Y = np.conj(np.fft.fft(y, n))
    f = np.fft.fftfreq(len(t), dt)

    idx_range = (f >= 0)
    #return array([f, Y]).T
    return array([f[idx_range], Y[idx_range]]).T


def do_ifft(data_fd, hermitian=True):
    freqs, y_fd = data_fd[:, 0].real, data_fd[:, 1]

    y_fd = nan_to_num(y_fd)
    #print(y_fd.shape)
    #print(y_fd)
    if hermitian:
        y_fd = np.concatenate((np.conj(y_fd), np.flip(y_fd[1:])))
        #y_fd[0] *= 0
        """
        * ``a[0]`` should contain the zero frequency term,
        * ``a[1:n//2]`` should contain the positive-frequency terms,
        * ``a[n//2 + 1:]`` should contain the negative-frequency terms, in
          increasing order starting from the most negative frequency.
        """

    y_td = ifft(y_fd)
    df = np.mean(np.diff(freqs))
    n = len(y_td)

    t = np.arange(0, n) / (n * df)
    # t += 885

    #y_td = np.flip(y_td)
    #y_td = np.roll(y_td, 1)

    return array([t, y_td]).T

def load_data():
    data_dir = Path(ROOT_DIR / "data" / "T-Sweeper_and_TeraFlash" / "Lackierte Keramik" / "Puls (TeraFlash)")

    ref_file = data_dir / "2020_02_20_Metallplaetchen_Ref2_1000AVG.csv"

    ref_td = pd.read_csv(ref_file).values

    t = ref_td[:, 0] - ref_td[0, 0]

    ref_td = np.array([t, ref_td[:, 2]]).T

    return ref_td

ref_td = load_data()

ref_fd = do_fft(ref_td)
ref_td_new = do_ifft(ref_fd)

plt.figure("Time domain")
plt.plot(ref_td[:, 0], ref_td[:, 1], label=f"Ref")
plt.plot(ref_td_new[:, 0], ref_td_new[:, 1], label=f"test_t_ref")
plt.xlabel("Time (ps)")
plt.ylabel("Amplitude (nA)")
plt.legend()

plt.show()
