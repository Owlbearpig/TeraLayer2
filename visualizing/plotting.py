import numpy as np
import matplotlib.pyplot as plt
from functions import format_data, load_files, multir_numba
from consts import default_mask, um, c0, GHz

def plot_only_y():
    """
    for picking indices manually...
    :return: None
    """
    lam, R = format_data()

    plt.plot(R, label='measurement')

    plt.ylim((0, 1.1))
    plt.ylabel('$r^2$ (arb. units)')
    plt.legend()
    plt.show()

def plot_result(p, fun=multir_numba, mask=default_mask):
    from results import d_best
    lam, R = format_data(sample_file_idx=0)

    plt.title(f'fit: {np.round(p*um, 2)} \n best fit: {np.round(d_best*um, 2)}')
    plt.plot(lam / 1e-3, R, label='measurement')
    plt.plot(lam[mask] / 1e-3, R[mask], 'o', color='red')
    plt.plot(lam / 1e-3, fun(lam, p), label='fit')
    plt.plot(lam / 1e-3, multir_numba(lam, d_best), label='best fit (scipy/matlab LM-algo)')

    plt.xlim((0, 2))
    plt.ylim((0, 1.1))
    plt.xlabel('THZ-Wavelenght (mm)')
    plt.ylabel('$r^2$ (arb. units)')
    plt.legend()
    plt.show()


def plot_measured_ampl(sample_idx=0):
    f, r, b, s = load_files(sample_file_idx=sample_idx)
    lam, R = format_data()

    plt.plot(lam / 1e-3, r, label='reference')
    plt.plot(lam / 1e-3, b, label='background')
    plt.plot(lam / 1e-3, s, label=f'sample Kopf_1x_{sample_idx+1:04}')
    plt.xlim((0, 2))
    plt.ylim((0, 1.1))
    plt.xlabel('THZ-Wavelenght (mm)')
    plt.ylabel('$r$ (arb. units)')
    plt.legend()
    plt.show()


def plot_R(lam, R):
    plt.figure()
    plt.plot(lam / 1e-3, R, label='measurement')
    plt.xlim((0, 2))
    plt.ylim((0, 1.1))
    plt.xlabel('THZ-Wavelenght (mm)')
    plt.ylabel('$r^2$ (arb. units)')
    plt.legend()
    plt.show()


def plot_measured_phase(sample_idx=0):
    f, r, b, s = load_files(sample_file_idx=sample_idx, data_type='phase')
    f, r, b, s = f[default_mask], r[default_mask], b[default_mask], s[default_mask]
    data_slice = (f < 850*GHz)*(f > 250*GHz)
    #f, r, b, s = f[data_slice], np.unwrap(r[data_slice]), np.unwrap(b[data_slice]), np.unwrap(s[data_slice])
    f, r, b, s = f[data_slice], (r[data_slice]), (b[data_slice]), (s[data_slice])
    lam = c0 / f

    plt.plot(f/GHz, r, label='reference')
    #plt.plot(lam / 1e-3, b, label='background')
    plt.plot(f/GHz, s, label=f'sample Kopf_1x_{sample_idx+1:04}')
    plt.plot(f/GHz, abs(r-s), label=f'abs(r-s)')
    #plt.xlim((0, 2))
    #plt.ylim((0, 1.1))
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('phase (rad)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_measured_ampl()
