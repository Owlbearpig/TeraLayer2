import matplotlib.pyplot as plt
from functions import (format_data, load_files, multir_numba, find_files,
                       map_maskname, format_data_avg, get_phase_measured, f_axis)
from model.multir import multir
from consts import *


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


def plot_result(p, fun=multir, mask=default_mask, sample_file_idx=0, x_lim=(0, 2), fun_comparison=None, use_avg=False):
    from results import d_best
    if use_avg:
        lam, R = format_data_avg()
    else:
        lam, R = format_data(sample_file_idx=sample_file_idx)

    plt.title(f'fit: {np.round(p * um, 2)} \n best fit: {np.round(d_best * um, 2)}')
    plt.plot(lam / 1e-3, R, label='Measurement')
    plt.plot(lam[mask] / 1e-3, R[mask], 'o', color='red')
    plt.plot(lam / 1e-3, fun(lam, p), label='Fit')
    plt.plot(lam / 1e-3, multir_numba(lam, d_best), label='best fit (scipy/matlab LM-algo)')
    if fun_comparison:
        plt.plot(lam / 1e-3, fun_comparison(lam, p), label='Fit (comparison func.)')

    plt.xlim(x_lim)
    plt.ylim((0, 1.1))
    plt.xlabel('THZ-Wavelenght (mm)')
    plt.ylabel('$R_0$ (arb. units)')
    plt.legend()
    plt.show()


def plot_measured_ampl(sample_idx=0, x_axis='wl'):
    f = f_axis()
    r, b, s = load_files(sample_file_idx=sample_idx)
    lam, R = format_data()

    if x_axis == 'wl':
        x = lam / 1e-3
        plt.xlim((0, 1))
        plt.xlabel('THZ-Wavelength (mm)')
    else:
        x = f / GHz
        plt.xlim((0, 1000))
        plt.xlabel('THZ-Frequency (GHz)')

    plt.plot(x, -10 * np.log10(r), label='reference')
    plt.plot(x, -10 * np.log10(b), label='background')
    plt.plot(x, -10 * np.log10(s), label=f'sample Kopf_1x_{sample_idx + 1:04}')

    # plt.ylim((0, 1.1))
    plt.ylabel('Amplitude (dB)')
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


def plot_measured_phase(sample_idx=0, mask=None):
    """
    check commented out parts ... (unwrapping)
    :param sample_idx:
    :return:
    """
    f, r, b, s = get_phase_measured(sample_file_idx=sample_idx, mask=mask)

    f, r, b, s = f, np.unwrap(r), np.unwrap(b), np.unwrap(s)

    lam = c0 / f

    plt.plot(f / GHz, r, label='reference')
    # plt.plot(lam / 1e-3, b, label='background')
    plt.plot(f / GHz, s, label=f'sample Kopf_1x_{sample_idx + 1:04}')
    plt.plot(f / GHz, abs(r - s), label=f'abs(r-s)')
    # plt.xlim((0, 2))
    # plt.ylim((0, 1.1))
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('phase (rad)')
    plt.legend()
    plt.show()


def plot_thicknesses():
    optimization_results = find_files(optimization_results_dir, file_extension='npy')
    for result in optimization_results:
        method, mask_name = str(result.stem).split('-')
        if 'default' in mask_name:
            continue
        method = method.replace('_minimize_', '')
        mask = map_maskname(mask_name)
        min_f, max_f = mask[0], mask[-1]

        thicknesses = np.load(result)

        # bounds used (don't change these)
        d0 = array([50, 600, 50])
        lb = d0 - array([50, 50, 50])
        hb = d0 + array([50, 50, 50])
        plt.text(0, 300, f'Bounds: $(d_1,d_2,d_3)$:\n$d_0=${d0[0], d0[1], d0[2]} $\pm$ 50 ({mu_}m)')

        d1, d2, d3 = thicknesses[:, 0] * um, thicknesses[:, 1] * um, thicknesses[:, 2] * um
        plt.text(0, 500, fr'Avg. $d_1$: {round(np.mean(d1), 2)} $\pm$ {round(np.std(d1), 2)} ({mu_}m)')
        plt.text(0, 450, fr'Avg. $d_2$: {round(np.mean(d2), 2)} $\pm$ {round(np.std(d2), 2)} ({mu_}m)')
        plt.text(0, 400, fr'Avg. $d_3$: {round(np.mean(d3), 2)} $\pm$ {round(np.std(d3), 2)} ({mu_}m)')

        mask_str = f'{mask} (GHz)' if (len(mask) < 10) else ''
        plt.title(f'{method}, range: {min_f, max_f} (GHz), rez: {len(mask)}\n{mask_str}')
        plt.plot(thicknesses[:, 0] * um, label='$d_1$')
        plt.plot(thicknesses[:, 1] * um, label='$d_2$')
        plt.plot(thicknesses[:, 2] * um, label='$d_3$')
        plt.ylabel(f'Thickness ({mu_}m)')
        plt.xlabel('Measurement idx')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    p = array([166.66331658291458, 497.98994974874375, 553.2110552763819]) * um_to_m
    #plot_result(p, mask=custom_mask_420, sample_file_idx=10, x_lim=(0, 1), use_avg=False)
    plot_measured_phase(sample_idx=10)
    # plot_thicknesses()
    plot_measured_ampl(sample_idx=10, x_axis="g")
