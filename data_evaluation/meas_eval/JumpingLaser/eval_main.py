from parse_data import get_all_measurements, SystemEnum, MeasTypeEnum, SamplesEnum, Measurement, ModelMeasurement
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button, Slider
from helpers import plt_show
from meas_eval.mpl_settings import mpl_style_params, result_dir
from consts import thea, c_thz
from tmm_package import coh_tmm_slim_unsafe
from typing import List, Union
from samples import Sample
from functions import do_ifft

np.set_printoptions(precision=2)

all_measurements = get_all_measurements(add_model_measurements=True)

ref_measurements = [meas for meas in all_measurements if meas.meas_type == MeasTypeEnum.Reference]
bkg_measurements = [meas for meas in all_measurements if meas.meas_type == MeasTypeEnum.Background]
sam_measurements = [meas for meas in all_measurements if meas.meas_type == MeasTypeEnum.Sample]


def find_nearest_meas(meas1: Measurement, meas_list: List[Measurement]):
    if meas1.system == SystemEnum.Model:
        return meas1

    all_meas_same_system = [meas2 for meas2 in meas_list if meas1.system == meas2.system]
    abs_time_diffs = []
    for meas2 in all_meas_same_system:
        abs_time_diff = np.abs(meas2.time_diff(meas1))
        abs_time_diffs.append((abs_time_diff, meas2))

    sorted_time_diffs = sorted(abs_time_diffs, key=lambda x: x[0])

    closest_meas = sorted_time_diffs[0][1]
    if meas1.system != SystemEnum.TSweeper:
        closest_meas = sorted_time_diffs[0][1]

    return closest_meas


def plot_all_sweeps(measurements, freq_idx=2):
    for system in SystemEnum:
        if system == SystemEnum.TSweeper:
            continue
        print(system.name)
        meas_same_system = [meas for meas in measurements if (system == meas.system) and (meas.n_sweeps == 1)]

        fig_num = f"{system.name}"
        fig, (ax0, ax1) = plt.subplots(2, 1, num=fig_num)
        ax0.set_title(f"{system.name}")
        ax1.set_xlabel("Meas idx")
        ax0.set_ylabel("Amplitude (dB)")
        ax1.set_ylabel("Phase (rad)")
        for meas in meas_same_system:
            amp_db, phase = 20 * np.log10(meas.amp), meas.phase
            print(meas)
            print(f"Freqs: {meas.freq}")
            print(f"Direct mean amp.: {np.mean(amp_db, axis=0)}±{np.std(amp_db, axis=0)}")
            print(f"Direct mean phase: {np.mean(phase, axis=0)}±{np.std(phase, axis=0)}")
            for i, freq in enumerate(meas.freq):
                if i != freq_idx:
                    continue
                ax0.plot(amp_db[:, i], label=f"{meas.name} {meas.timestamp} {np.round(freq, 2)} THz")
                ax1.plot(phase[:, i], label=f"{meas.name} {meas.timestamp} {np.round(freq, 2)} THz")
        print()


def fix_r_phi_sign(meas: Measurement):
    if meas.n_sweeps == 1:
        print(f"Skipping {meas}, #sweeps: {meas.n_sweeps}")
        return

    meas_tsweeper = None
    for meas_ in sam_measurements:
        if (meas.sample == meas_.sample) and (meas_.system == SystemEnum.TSweeper):
            meas_tsweeper = meas_
            break

    if not meas_tsweeper:
        print(f"No TSweeper measurement for {meas}")
        return

    for freq_idx, freq in enumerate(meas.freq):
        tsweeper_r_avg = meas_tsweeper.r_avg[np.argmin(np.abs(meas_tsweeper.freq - freq))]
        tsweeper_r_phi_avg, meas_r_phi_avg = np.angle(tsweeper_r_avg), np.angle(meas.r_avg[freq_idx])

        if np.abs(tsweeper_r_phi_avg - meas_r_phi_avg) <= np.abs(tsweeper_r_phi_avg - -meas_r_phi_avg):
            meas.r_avg[freq_idx] = np.abs(meas.r_avg[freq_idx]) * np.exp(1j * meas_r_phi_avg)
        else:
            meas.r_avg[freq_idx] = np.abs(meas.r_avg[freq_idx]) * np.exp(-1j * meas_r_phi_avg)


def shift_freq_axis(sam_meas_: Measurement, ref_meas_: Measurement):
    shifts = {SamplesEnum.ampelMannRight: 0.010, SamplesEnum.fpSample5ceramic: -0 * 0.010,
              SamplesEnum.fpSample2: 0}
    try:
        shift = shifts[sam_meas_.sample]
    except KeyError:
        shift = 0

    sam_meas_.freq += shift
    ref_meas_.freq += shift


def fix_phase_slope(sam_meas_: Measurement):
    if sam_meas_.system != SystemEnum.TSweeper:
        return
    pulse_shifts = {SamplesEnum.fpSample5ceramic: 0.24, SamplesEnum.fpSample5Plastic: 0.39,
                    SamplesEnum.fpSample2: 0.24}
    try:
        pulse_shift = pulse_shifts[sam_meas_.sample]
    except KeyError:
        pulse_shift = 0

    phase_correction = -2 * np.pi * sam_meas_.freq * pulse_shift

    sam_meas_.phase += phase_correction
    sam_meas_.phase_avg += phase_correction


def fix_phase_offset(sam_meas_: Measurement):
    if sam_meas_.system != SystemEnum.PIC:
        return

    phase_offsets = {SamplesEnum.fpSample2: 0.424}
    try:
        offset = phase_offsets[sam_meas_.sample]
    except KeyError:
        offset = 0

    amp, phi = np.abs(sam_meas_.r_avg), np.angle(sam_meas_.r_avg)
    sam_meas_.r_avg = amp * np.exp(1j * (phi + offset))


def calc_sample_refl_coe(sample_enum: SamplesEnum):
    sample_meas = [meas for meas in all_measurements if meas.sample == sample_enum]
    for sam_meas in sample_meas:
        if sam_meas.system == SystemEnum.Model:
            sam_meas.simulate_sam_measurement()
            continue

        ref_meas = find_nearest_meas(sam_meas, ref_measurements)
        shift_freq_axis(sam_meas, ref_meas)
        fix_phase_slope(sam_meas)

        amp_ref, phi_ref = ref_meas.amp, ref_meas.phase
        amp_ref_avg, phi_ref_avg = ref_meas.amp_avg, ref_meas.phase_avg
        amp_sam, phi_sam = sam_meas.amp, sam_meas.phase
        amp_sam_avg, phi_sam_avg = sam_meas.amp_avg, sam_meas.phase_avg

        center = 0
        sign_ = -1
        if sam_meas.system == SystemEnum.TSweeper:
            sign_ = -1
            center = np.mean(np.unwrap(sign_ * (phi_sam_avg[500:1500] - phi_ref_avg[500:1500])))

        sam_meas.r = (amp_sam / amp_ref) * np.exp(sign_ * 1j * (phi_sam - phi_ref))
        sam_meas.r_avg = (amp_sam_avg / amp_ref_avg) * np.exp(sign_ * 1j * (phi_sam_avg - phi_ref_avg + center))

        fix_phase_offset(sam_meas)

    for sam_meas in sample_meas:
        # fix_r_phi_sign(sam_meas)
        pass


def plot_sample_refl_coe(sample_enum: SamplesEnum, less_plots: bool):
    sample_meas = [meas for meas in all_measurements if meas.sample == sample_enum]
    sample = sample_enum.value
    layer_cnt = sample.layers

    title = f"Avg. reflection coefficient. Sample: {sample_enum.name}"
    fig_r_avg_num = f"Avg. r {sample_enum.name}"
    fig_r_avg, (ax0_r_avg, ax1_r_avg) = plt.subplots(nrows=2, ncols=1, num=fig_r_avg_num)
    fig_r_avg.subplots_adjust(left=0.05 + 0.05 * layer_cnt, bottom=0.15 + 0.05 * layer_cnt)
    ax0_r_avg.set_title(title)
    ax0_r_avg.set_ylabel("Amplitude (dB)")
    ax1_r_avg.set_ylabel("Phase (rad)")
    ax1_r_avg.set_xlabel("Frequency (THz)")
    ax0_r_avg.set_xlim((-0.150, 2.1))
    ax1_r_avg.set_xlim((-0.150, 2.1))

    n_sliders, k_sliders, d_sliders = [], [], []
    n_slider_axes, k_slider_axes, d_slider_axes = [], [], []
    for layer_idx in range(layer_cnt):
        layer_n, layer_k = sample.ref_idx[layer_idx].real, sample.ref_idx[layer_idx].imag
        layer_n_min, layer_n_max = layer_n[0], layer_n[1]
        layer_k_min, layer_k_max = layer_k[0], layer_k[1]

        if np.isclose(layer_n_min, layer_n_max):
            layer_n_max += 0.001

        if np.isclose(layer_k_min, layer_k_max):
            layer_k_max += 0.0001

        n_slider_axes.append(fig_r_avg.add_axes([0.15, 0.10 - 0.05 * layer_idx, 0.25, 0.03]))
        n_slider = RangeSlider(ax=n_slider_axes[layer_idx],
                               label=f"n (layer {layer_idx})",
                               valmin=layer_n_min * 0.95,
                               valmax=layer_n_max * 1.05,
                               valinit=(layer_n_min, layer_n_max),
                               )
        n_sliders.append(n_slider)

        k_slider_axes.append(fig_r_avg.add_axes([0.60, 0.10 - 0.05 * layer_idx, 0.20, 0.03]))
        k_slider = RangeSlider(ax=k_slider_axes[layer_idx],
                               label=f"$\kappa$ (layer {layer_idx})",
                               valmin=0,
                               valmax=np.abs(layer_k_max) * 1.2,
                               valinit=(np.abs(layer_k_min), np.abs(layer_k_max)),
                               )
        k_sliders.append(k_slider)

        layer_thickness = sample.thicknesses[layer_idx]

        d_slider_axes.append(fig_r_avg.add_axes([0.03 + 0.04 * layer_idx, 0.20, 0.03, 0.60]))
        d_slider = Slider(ax=d_slider_axes[layer_idx],
                          label=f"d (layer {layer_idx})",
                          valmin=layer_thickness * 0.95,
                          valmax=layer_thickness * 1.05,
                          valinit=layer_thickness,
                          orientation='vertical',
                          )
        d_sliders.append(d_slider)

    resetax0 = fig_r_avg.add_axes([0.8, 0.010, 0.1, 0.04])
    reset_but = Button(resetax0, 'Reset', hovercolor='0.975')

    def reset0(event):
        for slider in (n_sliders + k_sliders + d_sliders):
            slider.reset()

    reset_but.on_clicked(reset0)

    mod_amp_scat, mod_phi_scat, mod_meas = None, None, None
    for sam_meas in sample_meas:
        ref_meas = find_nearest_meas(sam_meas, ref_measurements)

        amp_ref, phi_ref = ref_meas.amp, ref_meas.phase
        amp_sam, phi_sam = sam_meas.amp, sam_meas.phase

        if sam_meas.system == SystemEnum.TSweeper:
            ax0_r_avg.plot(sam_meas.freq, 20 * np.log10(np.abs(sam_meas.r_avg)),
                           label=sam_meas.system.name, c="grey")
            ax1_r_avg.plot(sam_meas.freq, np.angle(sam_meas.r_avg), label=sam_meas.system.name, c="grey")
        elif sam_meas.system == SystemEnum.Model:
            mod_amp_scat = ax0_r_avg.scatter(sam_meas.freq, 20 * np.log10(np.abs(sam_meas.r_avg)),
                                             label=sam_meas.system.name, c="black")
            mod_phi_scat = ax1_r_avg.scatter(sam_meas.freq, np.angle(sam_meas.r_avg),
                                             label=sam_meas.system.name, c="black")
            mod_meas = sam_meas
        else:
            ax0_r_avg.scatter(sam_meas.freq, 20 * np.log10(np.abs(sam_meas.r_avg)),
                              label=sam_meas.system.name, s=22, zorder=9)
            ax1_r_avg.scatter(sam_meas.freq, np.angle(sam_meas.r_avg), label=sam_meas.system.name, s=22, zorder=9)

        if sam_meas.n_sweeps != 1 and not less_plots:
            fig, (ax0, ax1) = plt.subplots(2, 1, num=str(ref_meas))
            ax0.set_title(f"{ref_meas}")
            ax1.set_xlabel("Meas idx")
            ax0.set_ylabel("Amplitude (dB)")
            ax1.set_ylabel("Phase (rad)")

            for i, freq in enumerate(ref_meas.freq):
                ax0.plot(20 * np.log10(amp_ref[:, i]), label=f"{np.round(freq, 2)} THz")
                ax1.plot(phi_ref[:, i], label=f"{np.round(freq, 2)} THz")

            fig, (ax0, ax1) = plt.subplots(2, 1, num=str(sam_meas))
            ax0.set_title(f"{sam_meas}")
            ax1.set_xlabel("Meas idx")
            ax0.set_ylabel("Amplitude (dB)")
            ax1.set_ylabel("Phase (rad)")

            for i, freq in enumerate(sam_meas.freq):
                ax0.plot(20 * np.log10(amp_sam[:, i]), label=f"{np.round(freq, 2)} THz")
                ax1.plot(phi_sam[:, i], label=f"{np.round(freq, 2)} THz")

        if (sam_meas.system == SystemEnum.TSweeper) and not less_plots:
            plt.figure("TSWeeper amp")
            plt.title("TSWeeper amp")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")
            plt.xlim((-0.150, 2.1))

            plt.plot(ref_meas.freq[:-2], 20 * np.log10(ref_meas.amp[:-2]), label=ref_meas)
            plt.plot(sam_meas.freq[:-2], 20 * np.log10(sam_meas.amp[:-2]), label=sam_meas)
            # plt.plot(bkg_meas.freq, np.log10(np.abs(bkg_meas.amp)), label="background")

            plt.figure("TSWeeper phase")
            plt.title("TSWeeper phase")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Phase (rad)")
            plt.xlim((-0.150, 2.1))

            plt.plot(ref_meas.freq[:-2], ref_meas.phase[:-2], label=ref_meas)
            plt.plot(sam_meas.freq[:-2], sam_meas.phase[:-2], label=sam_meas)
            # plt.plot(bkg_meas.freq, bkg_meas.phase, label="background")

        freq_idx = 2
        fig_num = f"r all meas. {sam_meas.sample.name}"
        title = f"Reflection coefficient. All meas. {sam_meas.sample.name} {np.round(sam_meas.freq[freq_idx], 2)} THz"

        if not plt.fignum_exists(fig_num):
            fig, (ax0, ax1) = plt.subplots(2, 1, num=fig_num)
            ax0.set_title(title)
            ax1.set_xlabel("Meas idx")
            ax0.set_ylabel("Amplitude (dB)")
            ax1.set_ylabel("Phase (rad)")
        else:
            fig = plt.figure(fig_num)
            ax0, ax1 = fig.get_axes()

        if sam_meas.n_sweeps != 1:
            r_amp_db, r_phi = 20 * np.log10(np.abs(sam_meas.r)), np.angle(sam_meas.r)

            print(sam_meas.system)
            print(f"Frequencies: {sam_meas.freq}")
            print(f"ref mean phase: {ref_meas.phase_avg}±{np.std(ref_meas.phase, axis=0)}")
            print(f"sample mean phase: {sam_meas.phase_avg}±{np.std(sam_meas.phase, axis=0)}")
            print(f"r direct mean Amp. {np.mean(r_amp_db, axis=0)}±{np.std(r_amp_db, axis=0)}")
            print(f"r direct mean phase: {np.mean(r_phi, axis=0)}±{np.std(r_phi, axis=0)}\n")
            for i, freq in enumerate(sam_meas.freq):
                if i != freq_idx:
                    continue
                ax0.plot(r_amp_db[:, i], label=f"{sam_meas} {np.round(freq, 2)} THz")
                ax1.plot(r_phi[:, i], label=f"{sam_meas} {np.round(freq, 2)} THz")

    def update(val):
        new_ref_idx = []
        for layer_idx_ in range(layer_cnt):
            n_min = n_sliders[layer_idx_].val[0] - 1j * k_sliders[layer_idx_].val[0]
            n_max = n_sliders[layer_idx_].val[1] - 1j * k_sliders[layer_idx_].val[1]
            new_ref_idx.append((n_min, n_max))

        sample.set_ref_idx(new_ref_idx)

        new_thicknesses = [d_slider_.val for d_slider_ in d_sliders]
        sample.set_thicknesses(new_thicknesses)

        mod_meas.simulate_sam_measurement(fast=True)
        freqs = mod_meas.freq
        new_amp, new_phi = 20 * np.log10(np.abs(mod_meas.r_avg)), np.angle(mod_meas.r_avg)

        mod_amp_scat.set_offsets(np.array([freqs, new_amp]).T)
        mod_phi_scat.set_offsets(np.array([freqs, new_phi]).T)
        fig_r_avg.canvas.draw_idle()

    for slider in n_sliders + k_sliders + d_sliders:
        slider.on_changed(update)

    plt_show(mpl, en_save=False)


def thickness_eval(sample_enum: SamplesEnum):
    sample_meas = [meas for meas in all_measurements if meas.sample == sample_enum]
    for meas in sample_meas:
        if meas.system in [SystemEnum.TSweeper, SystemEnum.Model]:
            continue

        if len(meas.sample.value.thicknesses) == 1:
            print(f"Evaluating: {meas}")
            single_layer_eval(meas)


def single_layer_eval(sam_meas_: Measurement):
    r_amp, r_phi = np.abs(sam_meas_.r_avg), np.angle(sam_meas_.r_avg)

    selected_freqs = sam_meas_.freq
    n = sam_meas_.sample.value.get_ref_idx(sam_meas_.freq)

    def calc_cost(p_):
        r_mod = np.zeros_like(selected_freqs, dtype=complex)
        for f_idx, freq in enumerate(selected_freqs):
            if f_idx in [0]:
                r_mod[f_idx] = sam_meas_.r_avg[f_idx]
                continue
            lam_vac = c_thz / freq
            if sam_meas_.sample.value.has_iron_core:
                d_ = np.array([np.inf, *p_, 10, np.inf], dtype=float)
            else:
                d_ = np.array([np.inf, *p_, np.inf], dtype=float)
            r_mod[f_idx] = -1 * coh_tmm_slim_unsafe("s", n[f_idx], d_, thea, lam_vac)

        r_mod_amp, r_mod_phi = np.abs(r_mod), np.angle(r_mod)

        amp_error = (r_amp - r_mod_amp) ** 2
        phi_error = (r_phi - r_mod_phi) ** 2

        """
        real_error = (r_mod.real - r_exp_real) ** 2
        imag_error = (r_mod.imag - r_exp_imag) ** 2

        return np.sum(real_error + imag_error)
        """

        return np.sum(amp_error + phi_error)

    d_truth = sam_meas_.sample.value.thicknesses[0]
    d_min, d_max = np.max([1, d_truth - 500]), d_truth + 500
    d1 = np.arange(int(d_min), int(d_max), 1, dtype=float)
    losses = []
    for d in d1:
        p = np.array([d])
        err = calc_cost(p)
        losses.append(err)

    plt.figure(sam_meas_.sample.name)
    plt.plot(d1, losses, label=sam_meas_)
    plt.xlabel("d1 (um)")
    plt.ylabel("Summed(Freq) residuals")
    print("Found minimum: ", d1[np.argmin(losses)])


if __name__ == '__main__':
    selected_sample = SamplesEnum.fpSample2

    new_rcparams = {"savefig.directory": result_dir / "GoodResults" / str(selected_sample.name)}
    mpl.rcParams = mpl_style_params(new_rcparams)

    calc_sample_refl_coe(selected_sample)
    plot_sample_refl_coe(selected_sample, less_plots=True)
    thickness_eval(selected_sample)

    plt_show(mpl, en_save=False)
