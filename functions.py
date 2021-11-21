from numpy import array
from consts import MHz, c0, data_dir
import pandas as pd
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
