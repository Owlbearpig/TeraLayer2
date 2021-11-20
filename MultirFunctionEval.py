import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from consts import ROOT_DIR, MHz, c0
from pathlib import Path
from multir import multir
import pandas as pd
import time

data_dir = Path(ROOT_DIR / 'matlab_enrique' / 'Data')


def read_csv(file_path):
    return array(pd.read_csv(file_path, usecols = [i for i in range(5)]))


r = read_csv(data_dir / 'ref_1000x.csv')
b = read_csv(data_dir / 'BG_1000.csv')
s = read_csv(data_dir / 'Kopf_1x' / 'Kopf_1x_0001')

f = r[235:-2, 0] * MHz

lam = c0 / f

rr = r[235:-2, 1] - b[235:-2, 1]
ss = s[235:-2, 1] - b[235:-2, 1]
T = ss / rr

R = T**2

ni, nf, nn = 400, 640, 40
enes = np.arange(ni, nf, nn)

d = array([0.0000378283, 0.0006273254, 0.0000378208])

t0 = time.perf_counter()
repeats = 1000
for _ in range(repeats):
    multir(d, lam[enes])

print(f'{1000*(time.perf_counter() - t0)/repeats} ms / func. eval')

plt.plot(lam/1e-3, R)
plt.plot(lam[enes]/1e-3, R[enes], 'o', color='red')
plt.plot(lam/1e-3, multir(d, lam))
plt.xlim((0, 2))
plt.ylim((0, 1.1))
plt.xlabel('THZ-Wavelenght (mm)')
plt.ylabel('$r^2$ (arb. units)')
plt.show()
