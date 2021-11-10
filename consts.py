from numpy import pi
from pathlib import Path
import os

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

rad = 180 / pi
thea=8*pi/180
a = 1
n = [1, 1.50, 2.8, 1.50, 1]

nm = 10**-9
MHz = 10**6
GHz = 10**9
THz = 10**12