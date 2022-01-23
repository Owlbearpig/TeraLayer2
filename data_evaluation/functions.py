from numpy import sum
from model.multir_numba import multir_numba
import pandas as pd
from consts import *
import time


def find_files(top_dir=ROOT_DIR, search_str='', file_extension=''):
    results = [Path(os.path.join(root, name))
               for root, dirs, files in os.walk(top_dir)
               for name in files if name.endswith(file_extension) and search_str in str(name)]
    return results


def read_csv(file_path):
    return array(pd.read_csv(file_path, usecols = [i for i in range(5)]))


def avg_runtime(fun, *args, **kwargs):
    repeats = 100
    t0 = time.perf_counter()
    for _ in range(repeats):
        fun(*args, **kwargs)

    print(f'{fun.__name__}: {1e6 * (time.perf_counter() - t0) / repeats} \u03BCs / func. eval. ({repeats} calls)')


def load_files(sample_file_idx=0, data_type='amplitude'):
    slice_0, slice_1 = 235, -2

    r = read_csv(data_dir / 'ref_1000x.csv')
    b = read_csv(data_dir / 'BG_1000.csv')
    s = read_csv(data_dir / 'Kopf_1x' / f'Kopf_1x_{sample_file_idx:04}')

    f = r[slice_0:slice_1, 0] * MHz

    if data_type == 'amplitude':
        return f, r[slice_0:slice_1, 1], b[slice_0:slice_1, 1], s[slice_0:slice_1, 1]
    else:
        return f, s[slice_0:slice_1, 4], b[slice_0:slice_1, 2], s[slice_0:slice_1, 2]


def format_data(mask=None, sample_file_idx=0):
    f, r, b, s = load_files(sample_file_idx)

    lam = c0 / f
    rr = r - b
    ss = s - b
    reflectance = ss / rr

    reflectivity = (reflectance ** 2).real  # imaginary part should be 0

    if mask is not None:
        print(f[mask]/GHz, 'Selected frequencies (GHz)')
        return lam[mask], reflectivity[mask]
    else:
        return lam, reflectivity


def residuals(p, fun, x, y0):
    return (fun(x, p) - y0)**2


# could be a wrapper
def weighted_residuals(p, fun, x, y0, w):
    return w*residuals(p, fun, x, y0)


def calc_loss(p, **kwargs):
    lam, R0 = format_data(**kwargs)
    return sum((R0 - multir_numba(lam, p))**2)


def calc_full_loss(p, **kwargs):
    """
    calculates sum of squared residuals over full wl range
    """
    return calc_loss(p, mask=full_range_mask, **kwargs)


def calc_scipy_loss(p):
    return calc_loss(p)/len(p)


def map_maskname(mask):
    mask_map = {'custom_mask_420': custom_mask_420,
                'default_mask': default_mask,
                'full_range_mask_new': full_range_mask_new,
                }
    return mask_map[mask]


if __name__ == '__main__':
    from consts import wide_mask
    sample_idx = 10
    lam_w, R_w = format_data(wide_mask, sample_file_idx=sample_idx)
    print(lam_w, R_w)

    lam, R = format_data(default_mask, sample_file_idx=sample_idx)
    print(lam, R)

    lam, R = format_data(custom_mask_420, sample_file_idx=sample_idx)
    print(lam, R)