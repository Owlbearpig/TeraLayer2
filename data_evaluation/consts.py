from numpy import pi, array, round, sqrt, sign, cos, sin, exp, array, arcsin, conj, sum, outer
from scipy.special import factorial
from pathlib import Path
import numpy as np
import os
from scipy.constants import c as c0

Omega_, Delta_, sigma_, mu_, epsilon_, degree_ = '\u03BC', '\u0394', '\u03C3', '\u03BC', '\u03B5', '\u00B0'

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
data_dir = Path(ROOT_DIR / 'matlab_enrique' / 'Data')
optimization_results_dir = Path(ROOT_DIR / 'measurementComparisonResults')

data_file_cnt = 100

rad = 180 / pi
thea = 8 * pi / 180
a = 1
n = [1, 1.50, 2.8, 1.50, 1]

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
full_range_mask = np.arange(250, 1000, 1)
full_range_mask_new = np.arange(420, 1000, 1)  # based on plot of reference and background, big water line at 1 THz
full_range_mask_new_low_rez = np.arange(450, 1000, 100)
custom_mask_420 = array([420, 520, 650, 800, 850, 950])
high_freq_mask = np.arange(250, 600, 1)
high_freq_mask_low_rez = np.arange(760, 1000, 40)
high_freq_mask_low_rez2 = np.arange(640, 920, 40)
