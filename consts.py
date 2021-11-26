from numpy import pi
from pathlib import Path
import numpy as np
import os
from scipy.constants import c as c0

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
data_dir = Path(ROOT_DIR / 'matlab_enrique' / 'Data')

rad = 180 / pi
thea = 8*pi/180
a = 1
n = [1, 1.50, 2.8, 1.50, 1]

nm = 10**9
um = 10**6
MHz = 10**6
GHz = 10**9
THz = 10**12

um_to_m = 1/um

ni, nf, nn = 400, 640, 40  # 0-4500

default_mask = np.arange(ni, nf, nn)
default_mask_hr = np.arange(ni, nf, 1)  # default mask high rez

wide_mask = np.arange(300, 640, 40)
custom_mask = np.array([190, 195, 203, 210, 227, 240, 256, 260, 269, 280, 304,
                        345, 370, 380, 400, 422, 430, 459, 480, 501, 415, 538, 560,
                        580, 600, 628, 640, 656])
