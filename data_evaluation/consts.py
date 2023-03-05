from numpy import (pi, array, round, sqrt, sign, cos, sin, exp, array,
                   arcsin, conj, sum, outer, ones, inf, zeros)
from numpy.random import uniform
from scipy.special import factorial
from pathlib import Path
import numpy as np
import os
from scipy.constants import c as c0

c_thz = c0 * 10**-6  # um / ps

cur_os = os.name

settings = {
    'data_range_idx': (234, -2)
}

Omega_, Delta_, sigma_, mu_, epsilon_, degree_ = '\u03BC', '\u0394', '\u03C3', '\u03BC', '\u03B5', '\u00B0'

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
data_dir = Path(ROOT_DIR / 'matlab_enrique' / 'Data')
optimization_results_dir = Path(ROOT_DIR / 'measurementComparisonResults')
hhi_data_dir = Path(ROOT_DIR / "data" / "T-Sweeper_and_TeraFlash" /
                    "Lackierte Keramik" / "CW (T-Sweeper)" / "Kopf_Ahmad_3")

if os.name != "posix":
    op_besteck_dir = Path(r"E:\measurementdata\TeraLayer2\OP-Besteck\Blaues Teil")
else:
    op_besteck_dir = Path(r"/home/alex/Data/TeraLayer2/OP-Besteck/Blaues Teil")

data_file_cnt = 100

rad = 180 / pi
# thea = 8 * pi / 180
thea = 8 * pi / 180
a = 1

"""
20 1/cm POM 0.0477
0.25 1/cm PTFE 0.0006
@1 THz
5 1/cm POM 0.0120
~0.25 1/cm PTFE 0.0006
@0.5 THz
"""

#n = [1, 1.50, 2.8, 1.50, 1]
#n = [1, 1.35+0.0006*1j, 1.68+0.0120*1j, 1, 1]
#n = [1, 1.35, 1.68, 1, 1]
n = [1, 1.50, 2.80, 1.50, 1]

nm = 10 ** 9
um = 10 ** 6
MHz = 10 ** 6
GHz = 10 ** 9
THz = 10 ** 12

um_to_m = 1 / um

# ni, nf, nn = 400, 640, 40  # 0-4500 # original data indices
ni, nf, nn = 400, 640, 40  # 0-4500

default_mask = np.arange(ni, nf, nn)
default_mask_hr = np.arange(ni, nf, 1)  # default mask high rez

wide_mask = np.arange(250, 1000, 40)
all_freqs_lowend = np.arange(0, 1000, 1)
full_range_mask = np.arange(250, 1000, 1)
full_range_mask_new = np.arange(420, 1000, 1)  # based on plot of reference and background, big water line at 1 THz
full_range_mask_new_low_rez = np.arange(450, 1000, 100)
custom_mask_420 = array([420, 520, 650, 800, 850, 950])
high_freq_mask = np.arange(250, 600, 1)
high_freq_mask_low_rez = np.arange(760, 1000, 40)
high_freq_mask_low_rez2 = np.arange(640, 920, 40)
new_mask = array([299, 350, 499, 599, 799, 950])
