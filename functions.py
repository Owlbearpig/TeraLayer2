from numpy import array
from consts import MHz, c0, data_dir
from multir_numba import multir_numba
import pandas as pd
import matplotlib.pyplot as plt
from consts import default_mask
import time


def read_csv(file_path):
    return array(pd.read_csv(file_path, usecols = [i for i in range(5)]))


def avg_runtime(fun, *args, **kwargs):
    repeats = 1000
    t0 = time.perf_counter()
    for _ in range(repeats):
        fun(*args, **kwargs)

    print(f'{fun.__name__}: {1000 * (time.perf_counter() - t0) / repeats} ms / func. eval. ({repeats} calls)')


def format_data(mask=None):
    r = read_csv(data_dir / 'ref_1000x.csv')
    b = read_csv(data_dir / 'BG_1000.csv')
    s = read_csv(data_dir / 'Kopf_1x' / 'Kopf_1x_0001')

    f = r[235:-2, 0] * MHz

    lam = c0 / f

    rr = r[235:-2, 1] - b[235:-2, 1]
    ss = s[235:-2, 1] - b[235:-2, 1]
    reflectance = ss / rr

    reflectivity = reflectance ** 2

    if mask is not None:
        return lam[mask], reflectivity[mask]
    else:
        return lam, reflectivity


def residuals(p, fun, x, y0):
    return (fun(x, p)-y0)**2


def calc_loss(p):
    lam, R = format_data(default_mask)
    return sum((residuals(p, multir_numba, lam, R)))


def calc_scipy_loss(p):
    lam, R = format_data(default_mask)
    return sum((residuals(p, multir_numba, lam, R))**2)/2


def plot(p, fun=multir_numba):
    from results import d_best
    lam, R = format_data()

    plt.plot(lam / 1e-3, R, label='measurement')
    plt.plot(lam[default_mask] / 1e-3, R[default_mask], 'o', color='red')
    plt.plot(lam / 1e-3, fun(lam, p), label='fit')
    plt.plot(lam / 1e-3, multir_numba(lam, d_best), label='best fit (scipy/matlab LM-algo)')
    plt.xlim((0, 2))
    plt.ylim((0, 1.1))
    plt.xlabel('THZ-Wavelenght (mm)')
    plt.ylabel('$r^2$ (arb. units)')
    plt.legend()
    plt.show()
