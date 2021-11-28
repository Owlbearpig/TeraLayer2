from numpy import array
from consts import MHz, c0, data_dir
from model.multir_numba import multir_numba
import pandas as pd
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


def load_files(sample_file_idx=0):
    slice_0, slice_1 = 235, -2

    r = read_csv(data_dir / 'ref_1000x.csv')
    b = read_csv(data_dir / 'BG_1000.csv')
    s = read_csv(data_dir / 'Kopf_1x' / f'Kopf_1x_{sample_file_idx+1:04}')

    f = r[slice_0:slice_1, 0] * MHz

    return f, r[slice_0:slice_1, 1], b[slice_0:slice_1, 1], s[slice_0:slice_1, 1]


def format_data(mask=None, sample_file_idx=0):
    f, r, b, s = load_files(sample_file_idx)

    lam = c0 / f

    rr = r - b
    ss = s - b
    reflectance = ss / rr

    reflectivity = reflectance ** 2

    if mask is not None:
        return lam[mask], reflectivity[mask]
    else:
        return lam, reflectivity


def residuals(p, fun, x, y0):
    return (fun(x, p)-y0)**2


# could be a wrapper
def weighted_residuals(p, fun, x, y0, w):
    return w*residuals(p, fun, x, y0)


def calc_loss(p):
    lam, R = format_data(default_mask)
    return sum((residuals(p, multir_numba, lam, R)))


def calc_scipy_loss(p):
    lam, R = format_data(default_mask)
    return sum((residuals(p, multir_numba, lam, R))**2)/2


if __name__ == '__main__':
    from consts import wide_mask
    lam_w, R_w = format_data(wide_mask)
    print(lam_w, R_w)

    lam, R = format_data(default_mask)
    print(lam, R)