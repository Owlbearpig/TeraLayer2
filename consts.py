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

ni, nf, nn = 400, 640, 40
default_mask = np.arange(ni, nf, nn)
