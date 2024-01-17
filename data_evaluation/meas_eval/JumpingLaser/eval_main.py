from parse_data import get_all_measurements, SystemEnum, MeasTypeEnum, SamplesEnum
import numpy as np
import matplotlib.pyplot as plt
from helpers import plt_show
from meas_eval.mpl_settings import mpl_style_params
from consts import thea, c_thz
from tmm_package import coh_tmm_slim_unsafe

mpl_style_params()

np.set_printoptions(precision=2)

all_measurements = get_all_measurements()

ref_measurements = [meas for meas in all_measurements if meas.meas_type == MeasTypeEnum.Reference]
bkg_measurements = [meas for meas in all_measurements if meas.meas_type == MeasTypeEnum.Background]


def find_nearest_ref(meas):
    refs_same_system = [ref for ref in ref_measurements if ref.system == meas.system]
    abs_time_diffs = []
    for ref_meas in refs_same_system:
        abs_time_diff = np.abs(ref_meas.time_diff(meas))
        abs_time_diffs.append((abs_time_diff, ref_meas))

    sorted_refs = sorted(abs_time_diffs, key=lambda x: x[0])
    closest_ref = sorted_refs[0][1]
    if meas.system != SystemEnum.TSweeper:
        closest_ref = sorted_refs[0][1]

    return closest_ref


def plot_measurements(measurements, freq_idx=2):
    for system in SystemEnum:
        if system == SystemEnum.TSweeper:
            continue
        print(system.name)
        meas_same_system = [meas for meas in measurements if system == meas.system]

        fig_num = f"{system.name}"
        fig, (ax0, ax1) = plt.subplots(2, 1, num=fig_num)
        ax0.set_title(f"{system.name}")
        ax1.set_xlabel("Meas idx")
        ax0.set_ylabel("Amplitude (dB)")
        ax1.set_ylabel("Phase (rad)")
        for meas in meas_same_system:
            amp, phase = np.log10(np.abs(meas.amp)), meas.phase
            print(meas)
            print(f"Freqs: {meas.freq}")
            print(f"Mean Amp. {np.mean(amp, axis=0)}±{np.std(amp, axis=0)}")
            print(f"Mean phase: {np.mean(phase, axis=0)}±{np.std(phase, axis=0)}")
            for i, freq in enumerate(meas.freq):
                if i != freq_idx:
                    continue
                ax0.plot(amp[:, i], label=f"{meas.name} {meas.timestamp} {np.round(freq, 2)} THz")
                ax1.plot(phase[:, i], label=f"{meas.name} {meas.timestamp} {np.round(freq, 2)} THz")
        print()


def refl_coe_plot(measurements):
    for meas in measurements:
        ref_meas = find_nearest_ref(meas)
        meas.r_exp_car_avg = meas.data_car_avg / ref_meas.data_car_avg
        meas.r_exp_car = meas.data_car / ref_meas.data_car

        if meas.system == SystemEnum.WaveSourcePicFreq:
            fig, (ax0, ax1) = plt.subplots(2, 1, num=str(ref_meas))
            ax0.set_title(f"{ref_meas}")
            ax1.set_xlabel("Meas idx")
            ax0.set_ylabel("Amplitude (dB)")
            ax1.set_ylabel("Phase (rad)")

            for i, freq in enumerate(meas.freq):
                ax0.plot(np.log10(np.abs(ref_meas.amp[:, i])), label=f"{np.round(freq, 2)} THz")
                ax1.plot(ref_meas.phase[:, i], label=f"{np.round(freq, 2)} THz")

            fig, (ax0, ax1) = plt.subplots(2, 1, num=str(meas))
            ax0.set_title(f"{meas}")
            ax1.set_xlabel("Meas idx")
            ax0.set_ylabel("Amplitude (dB)")
            ax1.set_ylabel("Phase (rad)")

            for i, freq in enumerate(meas.freq):
                ax0.plot(np.log10(np.abs(meas.amp[:, i])), label=f"{np.round(freq, 2)} THz")
                ax1.plot(meas.phase[:, i], label=f"{np.round(freq, 2)} THz")
            print(f"sample mean phase: {np.mean(meas.phase, axis=0)}±{np.std(meas.phase, axis=0)}")

        if meas.system == SystemEnum.TSweeper:
            plt.figure("TSWeeper amp")
            plt.title("TSWeeper amp")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")
            plt.xlim((-0.150, 2.1))
            plt.ylim((-7.50, 2.5))

            plt.plot(ref_meas.freq, np.log10(np.abs(ref_meas.amp_avg)), label="reference")
            plt.plot(meas.freq, np.log10(np.abs(meas.amp_avg)), label="sample")

            plt.figure("TSWeeper phase")
            plt.title("TSWeeper phase")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Phase (rad)")
            plt.xlim((-0.150, 2.1))

            plt.plot(ref_meas.freq, ref_meas.phase_avg, label="reference")
            plt.plot(meas.freq, meas.phase_avg, label="sample")

        title = f"Avg. reflection coefficient. Sample: {meas.sample.name}"

        plt.figure(f"r_exp avg amp {meas.sample.name}")
        plt.title(title)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")
        plt.xlim((-0.150, 2.1))

        if meas.system == SystemEnum.TSweeper:
            plt.plot(meas.freq, np.log10(np.abs(meas.r_exp_car_avg)), label=meas.system.name, c="r")
        else:
            plt.scatter(meas.freq_OSA, np.log10(np.abs(meas.r_exp_car_avg)), label=meas.system.name, s=22, zorder=9)

        plt.figure(f"r_exp avg phase {meas.sample.name}")
        plt.title(title)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")
        plt.xlim((-0.150, 2.1))

        if meas.system == SystemEnum.TSweeper:
            plt.plot(meas.freq, np.angle(meas.r_exp_car_avg), label=meas.system.name, c="r")
        else:
            plt.scatter(meas.freq_OSA, np.angle(meas.r_exp_car_avg), label=meas.system.name, s=22, zorder=9)
            # phase_diff = meas.phase_avg - ref_meas.phase_avg
            # plt.scatter(meas.freq_OSA, phase_diff, label=f"Phase diff {meas.system.name}", s=22, zorder=9)

        freq_idx = 4
        fig_num = f"r_exp. All meas. {meas.sample.name}"
        title = f"Reflection coefficient. All meas. {meas.sample.name} {np.round(meas.freq[freq_idx], 2)}"

        if not plt.fignum_exists(fig_num):
            fig, (ax0, ax1) = plt.subplots(2, 1, num=fig_num)
            ax0.set_title(title)
            ax1.set_xlabel("Meas idx")
            ax0.set_ylabel("Amplitude (dB)")
            ax1.set_ylabel("Phase (rad)")
        else:
            fig = plt.figure(fig_num)
            ax0, ax1 = fig.get_axes()

        if meas.system != SystemEnum.TSweeper:
            amp, phase = np.log10(np.abs(meas.r_exp_car)), np.angle(meas.r_exp_car)
            print(f"r_exp mean Amp. {np.mean(amp, axis=0)}±{np.std(amp, axis=0)}")
            print(f"r_exp mean phase: {np.mean(phase, axis=0)}±{np.std(phase, axis=0)}\n")
            for i, freq in enumerate(meas.freq):
                if i != freq_idx:
                    continue
                ax0.plot(amp[:, i], label=f"{meas} {np.round(freq, 2)} THz")
                ax1.plot(phase[:, i], label=f"{meas} {np.round(freq, 2)} THz")


def single_layer_eval(measurements):
    for meas in measurements:
        if len(meas.sample.value.thicknesses) != 1:
            return

        if meas.system == SystemEnum.TSweeper:
            continue

        print(f"Evaluating: {meas}")
        ref_meas = find_nearest_ref(meas)

        meas.r_exp_car_avg = meas.data_car_avg / ref_meas.data_car_avg

        r_exp_real, r_exp_imag = meas.r_exp_car_avg.real, meas.r_exp_car_avg.imag
        r_exp_amp, r_exp_phi = np.abs(meas.r_exp_car_avg), np.angle(meas.r_exp_car_avg)

        selected_freqs = meas.freq
        one = np.ones_like(selected_freqs)
        n = np.array([one, *(meas.sample.value.ref_idx[:, np.newaxis] * one), one], dtype=float).T

        def calc_cost(p_):
            r_mod = np.zeros_like(selected_freqs, dtype=complex)
            for f_idx, freq in enumerate(selected_freqs):
                if f_idx in [2, 4]:
                    pass
                lam_vac = c_thz / freq
                d_ = np.array([np.inf, *p_, np.inf], dtype=float)
                r_mod[f_idx] = -1 * coh_tmm_slim_unsafe("s", n[f_idx], d_, thea, lam_vac)

            r_mod_amp, r_mod_phi = np.abs(r_mod), np.angle(r_mod)

            amp_error = (r_exp_amp - r_mod_amp) ** 2
            phi_error = (r_exp_phi - r_mod_phi) ** 2

            """
            real_error = (r_mod.real - r_exp_real) ** 2
            imag_error = (r_mod.imag - r_exp_imag) ** 2

            return np.sum(real_error + imag_error)
            """

            return np.sum(amp_error + phi_error)

        d1 = np.arange(1, 1000, 1, dtype=float)
        losses = []
        for d in d1:
            p = np.array([d])
            err = calc_cost(p)
            losses.append(err)

        plt.figure(meas.sample.name)
        plt.plot(d1, losses, label=meas)
        plt.xlabel("d1 (um)")
        plt.ylabel("Summed(Freq) residuals")
        print("Found minimum: ", d1[np.argmin(losses)])


if __name__ == '__main__':
    samples = [meas for meas in all_measurements if meas.sample == SamplesEnum.opToolBluePos2]

    # plot_system_meas(ref_measurements, freq_idx=4)
    refl_coe_plot(samples)
    single_layer_eval(samples)

    plt_show(plt)
