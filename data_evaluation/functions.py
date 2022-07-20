from numpy import sum
from data_evaluation.model.multir_numba import multir_numba
import pandas as pd
from data_evaluation.consts import *
import time
import string


def find_files(top_dir=ROOT_DIR, search_str='', file_extension=''):
    results = [Path(os.path.join(root, name))
               for root, dirs, files in os.walk(top_dir)
               for name in files if name.endswith(file_extension) and search_str in str(name)]
    return results


def read_csv(file_path):
    return array(pd.read_csv(file_path, usecols=[i for i in range(5)]))


def avg_runtime(fun, *args, **kwargs):
    repeats = 100
    t0 = time.perf_counter()
    for _ in range(repeats):
        fun(*args, **kwargs)

    print(f'{fun.__name__}: {1e6 * (time.perf_counter() - t0) / repeats} \u03BCs / func. eval. ({repeats} calls)')


def load_ref_file():
    r = read_csv(data_dir / 'ref_1000x.csv')
    slice_0, slice_1 = settings['data_range_idx']

    return r[:]
    #return r[slice_0:slice_1]


def f_axis():
    r = load_ref_file()
    return r[:, 0] * MHz


def lam_axis():
    return c0 / f_axis()


def load_files(sample_file_idx=0, data_type='amplitude'):
    slice_0, slice_1 = settings['data_range_idx']

    r = load_ref_file()
    b = read_csv(data_dir / 'BG_1000.csv')
    s = read_csv(data_dir / 'Kopf_1x' / f'Kopf_1x_{sample_file_idx:04}')

    if data_type == 'amplitude':
        #return r[:, 1], b[slice_0:slice_1, 1], s[slice_0:slice_1, 1]
        return r[:, 1], b[:, 1], s[:, 1]
    else:  # phase data columns, ref values are also present in each measurement file
        #return s[slice_0:slice_1, 4], b[slice_0:slice_1, 2], s[slice_0:slice_1, 2]
        return s[:, 4], b[:, 2], s[:, 2]


def format_data(mask=None, sample_file_idx=0, verbose=True):
    f = f_axis()
    r, b, s = load_files(sample_file_idx)

    lam = lam_axis()
    rr = r - b
    ss = s - b
    reflectance = ss / rr

    reflectivity = (reflectance ** 2).real  # imaginary part should be 0

    if mask is not None:
        if verbose:
            print(f[mask] / GHz, 'Selected frequencies (GHz)')
        return lam[mask], reflectivity[mask]
    else:
        return lam, reflectivity


def format_data_avg(mask=None, verbose=True):
    if verbose:
        print(f"using average of all {data_file_cnt} data files")

    s_all = []
    for sample_file_idx in range(data_file_cnt):
        _, _, s = load_files(sample_file_idx)
        s_all.append(s)

    s_avg = np.mean(np.array(s_all), axis=0)

    r, b, _ = load_files(0)
    lam, f = lam_axis(), f_axis()

    rr = r - b
    ss = s_avg - b
    reflectance = ss / rr

    reflectivity = (reflectance ** 2).real  # imaginary part should be 0

    if mask is not None:
        if verbose:
            print(f[mask] / GHz, 'Selected frequencies (GHz)')
        return lam[mask], reflectivity[mask]
    else:
        return lam, reflectivity


def residuals(p, fun, x, y0):
    return (fun(x, p) - y0) ** 2


# could be a wrapper
def weighted_residuals(p, fun, x, y0, w):
    return w * residuals(p, fun, x, y0)


def calc_loss(p, **kwargs):
    lam, R0 = format_data(**kwargs)
    return sum((R0 - multir_numba(lam, p)) ** 2)


def calc_full_loss(p, **kwargs):
    """
    calculates sum of squared residuals over full wl range
    """
    return calc_loss(p, mask=full_range_mask, **kwargs)


def calc_scipy_loss(p):
    return calc_loss(p) / len(p)


def map_maskname(mask):
    mask_map = {'custom_mask_420': custom_mask_420,
                'default_mask': default_mask,
                'full_range_mask_new': full_range_mask_new,
                }
    return mask_map[mask]


def get_phase_measured(sample_file_idx=0, mask=None):
    f = f_axis()
    r, b, s = load_files(sample_file_idx, data_type='phase')

    if mask is not None:
        return f[mask], r[mask], b[mask], s[mask]
    else:
        full_range = (f < 1000 * GHz) * (f > 250 * GHz)
        # full_range = (f > 250 * GHz)
        return f[full_range], r[full_range], b[full_range], s[full_range]


def get_full_measurement(sample_file_idx=0, mask=None, f_slice=None):
    f = f_axis()
    r_pha, b_pha, s_pha = load_files(sample_file_idx, data_type='phase')
    r_amp, b_amp, s_amp = load_files(sample_file_idx, data_type='amplitude')

    r_z, b_z, s_z = r_amp*np.exp(1j*r_pha), b_amp*np.exp(1j*b_pha), s_amp*np.exp(1j*s_pha)

    if mask is not None:
        return f[mask], r_z[mask], b_z[mask], s_z[mask]
    else:
        full_range = (f > -5000 * GHz)
        if f_slice is None:
            full_range = (f <= 1900 * GHz)*(f >= -100 * GHz)
        else:
            full_range = (f >= f_slice[0] * GHz)*(f <= f_slice[1] * GHz)
        return f[full_range], r_z[full_range], b_z[full_range], s_z[full_range]
        #return f[234:-2702], r_z[234:-2702], b_z[234:-2702], s_z[234:-2702]



def get_freq_idx(freqs):
    f = f_axis()
    res = []
    for freq in freqs:
        res.append(np.argmin(np.abs(f / GHz - freq)))

    return res


def do_fft(t, y):
    n = len(y)
    dt = np.float(np.mean(np.diff(t)))
    Y = np.fft.fft(y, n)
    f = np.fft.fftfreq(len(t), dt)
    idx_range = f > 0

    return f[idx_range], Y[idx_range]

def do_ifft(z):
    pass


def mult_2x2_matrix_chain(arr_in):
    # setup einsum_str (input shape)
    cnt = len(arr_in)
    s0 = string.ascii_lowercase + string.ascii_uppercase
    einsum_str = ''
    for i in range(cnt):
        einsum_str += s0[i] + s0[i + 1] + s0[cnt + 2] + ','

    # remove last comma
    einsum_str = einsum_str[:-1]
    # set output part of einsum_str
    einsum_str += '->' + s0[0] + s0[cnt] + s0[cnt + 2]

    M_out = np.einsum(einsum_str, *arr_in)

    return M_out


if __name__ == '__main__':
    from consts import wide_mask
    from visualizing.plotting import plot_R

    # sample_idx = 10
    # lam_w, R_w = format_data(wide_mask, sample_file_idx=sample_idx)
    # print(lam_w, R_w)

    # lam, R = format_data(default_mask, sample_file_idx=sample_idx)
    # print(lam, R)

    # lam, R = format_data(custom_mask_420, sample_file_idx=sample_idx)
    # print(lam, R)

    # lam, R_avg = format_data_avg()
    # plot_R(lam, R_avg)

    #f, r, b, s = get_phase_measured(sample_file_idx=10)

    # print(get_freq_idx([421., 521., 651., 801., 851., 951.]))
    #print(get_freq_idx([300, 351, 500, 600, 800, 951]))
    np.random.seed(123)

    cnt, m = 3, 1
    arr_in = np.random.random((cnt, 2, 2, m))
    A = arr_in[0, :, :, 0]
    B = arr_in[1, :, :, 0]
    C = arr_in[2, :, :, 0]
    out = np.dot(np.dot(A, B), C)
    #print(arr_in[0, :, :, 0])
    print(mult_2x2_matrix_chain(arr_in)[:, :, 0])
    print(out[:, :])