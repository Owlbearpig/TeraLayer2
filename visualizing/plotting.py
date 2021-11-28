import matplotlib.pyplot as plt
from functions import format_data, load_files, multir_numba
from consts import default_mask

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
    lam, R = format_data()

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


def plot_measurement(sample_idx=0):
    f, r, b, s = load_files(sample_file_idx=sample_idx)
    lam, R = format_data()

    #plt.plot(lam / 1e-3, r, label='reference')
    plt.plot(lam / 1e-3, b, label='background')
    #plt.plot(lam / 1e-3, s, label=f'sample Kopf_1x_000{sample_idx+1}')
    plt.xlim((0, 2))
    plt.ylim((0, 1.1))
    plt.xlabel('THZ-Wavelenght (mm)')
    plt.ylabel('$r^2$ (arb. units)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_only_y()
