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
from functools import partial
from functions import do_ifft, moving_average, std_err
from consts import selected_freqs as og_sel_freqs
from model.separation_idea.triple_layer_impl import triple_layer_impl


class Cost:
    def __init__(self, meas: Measurement):
        self.meas = meas
        self.selected_freqs = meas.freq
        self.n = meas.sample.value.get_ref_idx(meas.freq)
        self.r_avg = meas.r_avg
        self.r = meas.r

    def calc_cost(self, p_, verbose=False, sweep_idx_=None):
        if sweep_idx_ is None:
            r_exp = self.r_avg
        else:
            r_exp = self.r[sweep_idx_]

        r_real, r_imag = r_exp.real, r_exp.imag
        r_amp, r_phi = np.abs(r_exp), np.angle(r_exp)

        r_mod = np.zeros_like(self.selected_freqs, dtype=complex)
        for f_idx, freq in enumerate(self.selected_freqs):
            lam_vac = c_thz / freq
            if self.meas.sample.value.has_iron_core:
                d_ = np.array([np.inf, *p_, 10, np.inf], dtype=float)
            else:
                d_ = np.array([np.inf, *p_, np.inf], dtype=float)
            r_mod[f_idx] = -1 * coh_tmm_slim_unsafe("s", self.n[f_idx], d_, thea, lam_vac)

        r_mod_amp, r_mod_phi = np.abs(r_mod), np.angle(r_mod)

        # r_mod_amp[selected_freq < 0] = r_amp[selected_freq < 0]
        # r_mod_phi[selected_freq < 0] = r_phi[selected_freq < 0]

        # r_mod_amp[selected_freq > 0.8] = r_amp[selected_freq > 0.8]
        # r_mod_phi[selected_freq > 0.8] = r_phi[selected_freq > 0.8]

        cart_error = True
        if cart_error:
            real_error = (r_real - r_mod.real) ** 2
            imag_error = (r_imag - r_mod.imag) ** 2

            return np.sum(real_error + imag_error)
        else:
            amp_error = (r_amp - r_mod_amp) ** 2
            phi_error = (r_phi - r_mod_phi) ** 2

            if verbose:
                print(f"Amp error: {amp_error}")
                print(f"Phi error: {phi_error}")

            return np.sum(amp_error + phi_error)


class JumpingLaserEval:
    def __init__(self, new_options: dict):
        np.set_printoptions(precision=2)

        self.all_measurements = get_all_measurements(add_model_measurements=True)

        self.ref_measurements = [meas for meas in self.all_measurements if meas.meas_type == MeasTypeEnum.Reference]
        self.bkg_measurements = [meas for meas in self.all_measurements if meas.meas_type == MeasTypeEnum.Background]
        self.sam_measurements = [meas for meas in self.all_measurements if meas.meas_type == MeasTypeEnum.Sample]

        self.__options = self.set_options(new_options)

        self.plot_vars = {"truth_line_exists": False, "colors": ['red', 'green', 'blue', 'orange', 'purple']}

    def set_options(self, options_):
        selected_system = options_["selected_system"]
        if str(options_["selected_sweep"]).lower() == "random":
            n_sweeps_ = [meas for meas in self.sam_measurements if meas.system == selected_system][0].n_sweeps
            options_["selected_sweep"] = np.random.randint(0, n_sweeps_)

        __options = {"less_plots": False,
                     "debug_info": False,
                     }
        __options.update(options_)

        if options_["debug_info"]:
            print("DONT SAVE")  # Don't save plots with debug info on (not formatted correctly)

        return __options

    def find_nearest_meas(self, meas1: Measurement, meas_list: List[Measurement]):
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

    def plot_all_sweeps(self, measurements, freq_idx=2):
        for system in SystemEnum:
            if system == SystemEnum.TSweeper:
                continue

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

    def fix_r_phi_sign(self, meas: Measurement):
        if meas.n_sweeps == 1:
            print(f"Skipping {meas}, #sweeps: {meas.n_sweeps}")
            return

        meas_tsweeper = None
        for meas_ in self.sam_measurements:
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

    def shift_freq_axis(self, sam_meas_: Measurement, ref_meas_: Measurement):
        if sam_meas_.system == SystemEnum.PIC:
            shifts = {SamplesEnum.fpSample3: 0.006,
                      SamplesEnum.ampelMannRight: 0.0, SamplesEnum.fpSample5ceramic: 0.0,
                      SamplesEnum.fpSample2: 0.003, SamplesEnum.fpSample5Plastic: -0.006, SamplesEnum.fpSample6: 0.0,
                      SamplesEnum.bwCeramicBlackUp: 0.006,
                      SamplesEnum.ampelMannLeft: 0.000}
        elif sam_meas_.system == SystemEnum.TSweeper:
            shifts = {SamplesEnum.ampelMannRight: 0.0, SamplesEnum.fpSample5ceramic: 0.0,
                      SamplesEnum.fpSample2: 0.0, SamplesEnum.fpSample5Plastic: -0.006, SamplesEnum.fpSample6: 0.0}
        else:
            shifts = {}

        try:
            shift = shifts[sam_meas_.sample]
        except KeyError:
            shift = 0

        sam_meas_.freq += shift
        ref_meas_.freq += shift

    def fix_phase_slope(self, sam_meas_: Measurement):
        if sam_meas_.system == SystemEnum.TSweeper:
            pulse_shifts = {SamplesEnum.blueCube: 2.6, SamplesEnum.fpSample2: 0.24, SamplesEnum.fpSample3: 0.28,
                            SamplesEnum.fpSample5ceramic: 0.28, SamplesEnum.fpSample5Plastic: 0.39,
                            SamplesEnum.fpSample6: 0.1, SamplesEnum.bwCeramicWhiteUp: 0.20,
                            SamplesEnum.bwCeramicBlackUp: 0.26,
                            SamplesEnum.ampelMannRight: -0.02,
                            SamplesEnum.ampelMannLeft: 0.2,
                            SamplesEnum.opBlackPos1: 0.1,
                            SamplesEnum.opRedPos1: 0.2,  # not done
                            SamplesEnum.opBluePos1: 0.43}
        elif sam_meas_.system == SystemEnum.PIC:
            pulse_shifts = {SamplesEnum.fpSample2: 0.09, SamplesEnum.fpSample3: 0.09,
                            SamplesEnum.fpSample5ceramic: -0.16,
                            SamplesEnum.fpSample6: 0.2, SamplesEnum.bwCeramicBlackUp: 0.01,
                            SamplesEnum.bwCeramicWhiteUp: -0.069,
                            # SamplesEnum.ampelMannRight: -0.080,  # best
                            SamplesEnum.ampelMannRight: -0.080,  # -0.080
                            SamplesEnum.ampelMannLeft: 0.825,  # 0.825 best
                            SamplesEnum.opBlackPos1: -0.7,
                            SamplesEnum.opBluePos1: -0.95}
        elif sam_meas_.system == SystemEnum.WaveSource:
            pulse_shifts = {SamplesEnum.ampelMannLeft: 0.55,
                            }
        else:
            return

        try:
            pulse_shift = pulse_shifts[sam_meas_.sample]
        except KeyError:
            pulse_shift = 0

        sam_meas_.pulse_shift = pulse_shift

        phase_correction = -2 * np.pi * sam_meas_.freq * pulse_shift

        sam_meas_.phase += phase_correction
        sam_meas_.phase_avg += phase_correction

    def fix_tsweeper_offset(self, sam_meas_: Measurement):
        amp, phi = np.abs(sam_meas_.r_avg), np.angle(sam_meas_.r_avg)

        if sam_meas_.system == SystemEnum.TSweeper:
            offsets = {SamplesEnum.blueCube: 2.6, SamplesEnum.fpSample2: -np.mean(phi[540:1650]),
                       SamplesEnum.fpSample3: -1.37, SamplesEnum.fpSample5ceramic: -1.13,
                       SamplesEnum.fpSample5Plastic: 0.39,
                       SamplesEnum.fpSample6: -1.1, SamplesEnum.bwCeramicWhiteUp: 0.20}
        else:
            return

        try:
            offset = offsets[sam_meas_.sample]
        except KeyError:
            offset = 0

        sam_meas_.r_avg = amp * np.exp(1j * (phi + offset))

    def calc_sample_refl_coe(self):
        selected_sample = self.__options["selected_sample"]
        sample_meas = [meas for meas in self.all_measurements if meas.sample == selected_sample]
        for sam_meas in sample_meas:
            if sam_meas.system == SystemEnum.Model:
                sam_meas.simulate_sam_measurement()
                continue

            ref_meas = self.find_nearest_meas(sam_meas, self.ref_measurements)
            self.shift_freq_axis(sam_meas, ref_meas)
            self.fix_phase_slope(sam_meas)

            amp_ref, phi_ref = ref_meas.amp, ref_meas.phase
            amp_ref_avg, phi_ref_avg = ref_meas.amp_avg, ref_meas.phase_avg
            amp_sam, phi_sam = sam_meas.amp, sam_meas.phase
            amp_sam_avg, phi_sam_avg = sam_meas.amp_avg, sam_meas.phase_avg

            phase_sign_ = -1
            if sam_meas.system in [SystemEnum.PIC, SystemEnum.WaveSource]:
                phase_sign_ = 1

            phi_diff, phi_diff_avg = phi_sam - phi_ref, phi_sam_avg - phi_ref_avg
            phi_diff, phi_diff_avg = np.unwrap(phi_diff), np.unwrap(phi_diff_avg)

            amp_ratio = amp_sam / amp_ref
            amp_ratio_avg = amp_sam_avg / amp_ref_avg

            if sam_meas.system == SystemEnum.TSweeper:
                phi_diff_avg = moving_average(phi_diff_avg, window_size=2)
                amp_ratio_avg = moving_average(amp_ratio_avg, window_size=2)

            sam_meas.r = amp_ratio * np.exp(phase_sign_ * 1j * phi_diff)
            sam_meas.r_avg = amp_ratio_avg * np.exp(phase_sign_ * 1j * phi_diff_avg)

            self.fix_tsweeper_offset(sam_meas)

        for sam_meas in sample_meas:
            # fix_r_phi_sign(sam_meas)
            pass

        return sample_meas

    def plot_sample_refl_coe(self):
        selected_system = self.__options["selected_system"]
        selected_sample = self.__options["selected_sample"]
        selected_sweep_ = self.__options["selected_sweep"]
        less_plots = self.__options["less_plots"]
        shown_systems = [selected_system, SystemEnum.Model, SystemEnum.TSweeper]
        sample_meas = [meas for meas in self.all_measurements if
                       (meas.sample == selected_sample and meas.system in shown_systems)]

        sample = selected_sample.value
        layer_cnt = sample.layers

        if selected_sweep_ is None:
            title = f"Reflexionskoeffizient. Probe: {selected_sample.name}, Durchschnitt alle Aufnahmen"
            sweep_s = f"alle sweeps"
        else:
            title = f"Reflexionskoeffizient. Probe: {selected_sample.name}. Aufnahme {selected_sweep_}"
            sweep_s = f"sweep {selected_sweep_}"

        fig_r_num = f"r_{selected_sample.name}_{sweep_s}"
        fig_r, (ax0_r, ax1_r) = plt.subplots(nrows=2, ncols=1, num=fig_r_num)
        fig_r.subplots_adjust(left=0.05 + 0.05 * layer_cnt, bottom=0.15 + 0.05 * layer_cnt)
        ax0_r.set_title(title)
        ax0_r.set_ylabel("Amplitude (dB)")
        ax1_r.set_ylabel("Phase (rad)")
        ax1_r.set_xlabel("Frequenz (THz)")
        ax0_r.set_xlim((-0.150, 1.6))
        ax1_r.set_xlim((-0.150, 1.6))
        ax0_r.set_ylim((-40, 15))

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

            n_slider_axes.append(fig_r.add_axes([0.15, 0.10 - 0.05 * layer_idx, 0.16, 0.03]))
            label = f"Brechungsindex\nSchicht {layer_idx + 1}" if layer_idx == 0 else f"Schicht {layer_idx + 1}"
            n_slider = RangeSlider(ax=n_slider_axes[layer_idx],
                                   label=label,
                                   valmin=layer_n_min * 0.95,
                                   valmax=layer_n_max * 1.05,
                                   valinit=(layer_n_min, layer_n_max),
                                   )
            n_sliders.append(n_slider)

            k_slider_axes.append(fig_r.add_axes([0.60, 0.10 - 0.05 * layer_idx, 0.16, 0.03]))
            label = f"Extinktionskoeffizient\nSchicht {layer_idx + 1}" if layer_idx == 0 else f"Schicht {layer_idx + 1}"
            k_slider = RangeSlider(ax=k_slider_axes[layer_idx],
                                   label=label,
                                   valmin=0,
                                   valmax=np.abs(layer_k_max) * 1.2,
                                   valinit=(np.abs(layer_k_min), np.abs(layer_k_max)),
                                   )
            k_sliders.append(k_slider)

            layer_thickness = sample.thicknesses[layer_idx]

            d_slider_axes.append(fig_r.add_axes([0.03 + 0.05 * layer_idx, 0.20, 0.03, 0.60]))
            d_slider = Slider(ax=d_slider_axes[layer_idx],
                              label=f"Dicke\nSchicht\n{layer_idx + 1}",
                              valmin=layer_thickness * 0.25,
                              valmax=layer_thickness * 2,
                              valinit=layer_thickness,
                              orientation='vertical',
                              )
            d_sliders.append(d_slider)

        resetax0 = fig_r.add_axes([0.005, 0.01, 0.05, 0.04])
        reset_but = Button(resetax0, 'Reset', hovercolor='0.975')

        def reset0(event):
            for slider in (n_sliders + k_sliders + d_sliders):
                slider.reset()

        reset_but.on_clicked(reset0)

        mod_amp_line, mod_phi_line, mod_meas = None, None, None
        for sam_meas in sample_meas:
            ref_meas = self.find_nearest_meas(sam_meas, self.ref_measurements)

            legend_label = str(sam_meas.system.name)
            if self.__options["debug_info"]:
                legend_label += f"\nPulse shift {sam_meas.pulse_shift}"

            if sam_meas.system == SystemEnum.TSweeper:
                r = sam_meas.r_avg
                ax0_r.plot(sam_meas.freq, 20 * np.log10(np.abs(r)),
                           label=legend_label, c="grey", zorder=8)
                ax1_r.plot(sam_meas.freq, np.angle(r),
                           label=legend_label, c="grey", zorder=8)
            elif sam_meas.system == SystemEnum.Model:
                r = sam_meas.r_avg
                mod_amp_line, = ax0_r.plot(sam_meas.freq, 20 * np.log10(np.abs(r)),
                                           label=legend_label, c="black")
                mod_phi_line, = ax1_r.plot(sam_meas.freq, np.angle(r),
                                           label=legend_label, c="black")
                mod_meas = sam_meas
            else:
                r = sam_meas.r_avg if not selected_sweep_ else sam_meas.r[selected_sweep_]
                ax0_r.scatter(sam_meas.freq, 20 * np.log10(np.abs(r)),
                              label=legend_label, s=40, zorder=9, c="red")
                ax1_r.scatter(sam_meas.freq, np.angle(r),
                              label=legend_label, s=40, zorder=9, c="red")

            amp_ref, phi_ref = ref_meas.amp, ref_meas.phase
            amp_sam, phi_sam = sam_meas.amp, sam_meas.phase

            if sam_meas.n_sweeps != 1 and not less_plots:
                fig, (ax0, ax1) = plt.subplots(2, 1, num=str(ref_meas))
                ax0.set_title(f"{ref_meas}")
                ax1.set_xlabel("Aufnahme Index")
                ax0.set_ylabel("Amplitude (dB)")
                ax1.set_ylabel("Phase (rad)")

                for i, freq in enumerate(ref_meas.freq):
                    ax0.plot(20 * np.log10(amp_ref[:, i]), label=f"{np.round(freq, 2)} THz")
                    ax1.plot(phi_ref[:, i], label=f"{np.round(freq, 2)} THz")

                fig, (ax0, ax1) = plt.subplots(2, 1, num=str(sam_meas))
                ax0.set_title(f"{sam_meas}")
                ax1.set_xlabel("Aufnahme Index")
                ax0.set_ylabel("Amplitude (dB)")
                ax1.set_ylabel("Phase (rad)")

                for i, freq in enumerate(sam_meas.freq):
                    ax0.plot(20 * np.log10(amp_sam[:, i]), label=f"{np.round(freq, 2)} THz")
                    ax1.plot(phi_sam[:, i], label=f"{np.round(freq, 2)} THz")

            if (sam_meas.system == SystemEnum.TSweeper) and not less_plots:
                plt.figure("TSWeeper amp")
                plt.title("TSWeeper amp")
                plt.xlabel("Frequenz (THz)")
                plt.ylabel("Amplitude (dB)")
                plt.xlim((-0.150, 2.1))

                plt.plot(ref_meas.freq[:-2], 20 * np.log10(ref_meas.amp[:-2]), label=ref_meas)
                plt.plot(sam_meas.freq[:-2], 20 * np.log10(sam_meas.amp[:-2]), label=sam_meas)
                # plt.plot(bkg_meas.freq, np.log10(np.abs(bkg_meas.amp)), label="background")

                plt.figure("TSWeeper phase")
                plt.title("TSWeeper phase")
                plt.xlabel("Frequenz (THz)")
                plt.ylabel("Phase (rad)")
                plt.xlim((-0.150, 2.1))

                plt.plot(ref_meas.freq[:-2], ref_meas.phase[:-2], label=ref_meas)
                plt.plot(sam_meas.freq[:-2], sam_meas.phase[:-2], label=sam_meas)
                # plt.plot(bkg_meas.freq, bkg_meas.phase, label="background")

            if less_plots or sam_meas.n_sweeps <= 1:
                continue

            fig_num = f"r_all_sweeps_{sam_meas.system.name}_{sam_meas.sample.name}"
            freq_idx = None

            title = f"Reflexionskoeffizient {sam_meas.sample.name}. {sam_meas.system.name} alle Aufnahmen"
            if freq_idx is not None:
                title += f" {np.round(sam_meas.freq[freq_idx], 2)} THz"

            if not plt.fignum_exists(fig_num):
                fig, (ax0, ax1) = plt.subplots(2, 1, num=fig_num)
                ax0.set_title(title)
                ax1.set_xlabel("Aufnahme Index")
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
                    if freq_idx and (i != freq_idx):
                        continue
                    ax0.plot(r_amp_db[:, i], label=f"{np.round(freq, 2)} THz")
                    ax1.plot(r_phi[:, i], label=f"{np.round(freq, 2)} THz")

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

            mod_amp_line.set_data(freqs, new_amp)
            mod_phi_line.set_data(freqs, new_phi)
            fig_r.canvas.draw_idle()

        for slider in n_sliders + k_sliders + d_sliders:
            slider.on_changed(update)

        plt_show(mpl, en_save=save_plots)

    def thickness_eval(self):
        single_sweep_eval = self.__options["single_sweep_eval"]
        selected_sample = self.__options["selected_sample"]

        sample_meas = [meas for meas in self.all_measurements if meas.sample == selected_sample]
        ts_meas = [meas for meas in sample_meas if meas.system == SystemEnum.TSweeper][0]
        mod_meas = [meas for meas in sample_meas if meas.system == SystemEnum.Model][0]

        for meas in sample_meas:
            if len(meas.sample.value.thicknesses) == 3:
                print(f"Evaluating: {meas} (3 layers)")
                self.triple_layer_eval(meas, ts_meas, mod_meas, single_sweep_eval)

            if meas.system in [SystemEnum.TSweeper, SystemEnum.Model]:
                continue

            if len(meas.sample.value.thicknesses) == 1:
                print(f"Evaluating: {meas} (1 layer)")
                self.single_layer_eval(meas, ts_meas, mod_meas)

            if len(meas.sample.value.thicknesses) == 2:
                print(f"Evaluating: {meas} (2 layers)")
                self.double_layer_eval(meas, ts_meas, mod_meas, single_sweep_eval)

    def triple_layer_eval(self, sam_meas_: Measurement, ts_meas_: Measurement, mod_meas_: ModelMeasurement,
                          single_sweep_eval):
        selected_system = self.__options["selected_system"]
        if sam_meas_.system not in [selected_system, SystemEnum.TSweeper]:
            return

        triple_layer_impl(sam_meas_, ts_meas_, mod_meas_, self.__options)

    def double_layer_eval(self, sam_meas_: Measurement, ts_meas_: Measurement, mod_meas_: Measurement,
                          single_sweep_eval):
        selected_system = self.__options["selected_system"]
        if sam_meas_.system != selected_system:
            return

        ts_f_idx = []
        for selected_freq in sam_meas_.freq:
            ts_f_idx.append(np.argmin(np.abs(selected_freq - ts_meas_.freq)))

        r_amp_ts, r_phi_ts = np.abs(ts_meas_.r_avg[ts_f_idx]), np.angle(ts_meas_.r_avg[ts_f_idx])
        r_amp_truth, r_phi_truth = np.abs(mod_meas_.r_avg[ts_f_idx]), np.angle(mod_meas_.r_avg[ts_f_idx])
        r_real_truth, r_imag_truth = mod_meas_.r_avg[ts_f_idx].real, mod_meas_.r_avg[ts_f_idx].imag

        cost = Cost(sam_meas_)
        avg_cost = cost.calc_cost

        truth_d1, truth_d2 = sam_meas_.sample.value.thicknesses.astype(int)
        d1, d2 = np.arange(max(0, truth_d1 - 20), truth_d1 + 20, 1), np.arange(max(0, truth_d2 - 20), truth_d2 + 20, 1)

        plt.figure(str(sam_meas_.sample.name) + "_avg")
        img = np.zeros((len(d1), len(d2)))
        for i, d1_ in enumerate(d1):
            if (i % 25) == 0:
                print(f"{i}/{len(d1)}")
            for j, d2_ in enumerate(d2):
                img[i, j] = avg_cost(np.array([d1_, d2_]))

        plt.imshow(img,
                   extent=[d1[0], d1[-1], d2[0], d2[-1]],
                   origin="lower",
                   # interpolation='bilinear',
                   # cmap="plasma",
                   vmin=0, vmax=np.mean(img),
                   )
        plt.xlabel("Erste Schicht ($\mu$m)")
        plt.ylabel("Zweite Schicht ($\mu$m)")
        i, j = np.unravel_index(np.argmin(img), img.shape)
        print(f"Averaged sweeps. Global minimum {np.min(img)} at d1: {d1[i]} um, d2: {d2[j]} um")

        plt.figure(str(sam_meas_.sample.name) + "single_dim_avg")
        plt.title(f"Loss: [{truth_d1}, d2_] all sweeps averaged")
        d2_loss = []
        for d2_ in d2:
            d2_loss.append(avg_cost(np.array([truth_d1, d2_])))

        plt.plot(d2, d2_loss)
        plt.xlabel("Zweite Shicht ($\mu$m)")

        if single_sweep_eval or sam_meas_.system != selected_system:
            return

        n_sweeps = sam_meas_.n_sweeps
        sweeps = list(range(n_sweeps))
        results_d1, results_d2 = np.zeros((2, n_sweeps), dtype=float)
        min_losses = np.zeros(n_sweeps)
        for sweep_idx in sweeps:
            sweep_cost = partial(cost.calc_cost, sweep_idx_=sweep_idx)

            best_fit, best_fit_loss = (None, None), np.inf
            for i, d1_ in enumerate(d1):
                if (i % 25) == 0:
                    print(f"{i}/{len(d1)}")
                for j, d2_ in enumerate(d2):
                    p = np.array([d1_, d2_])
                    err = sweep_cost(p)

                    if err < best_fit_loss:
                        best_fit_loss = err
                        best_fit = (d1_, d2_)

            print(f"Found minimum: {best_fit} ({best_fit_loss}), sweep: {sweep_idx}")
            min_losses[sweep_idx] = best_fit_loss
            results_d1[sweep_idx], results_d2[sweep_idx] = best_fit

        mean_d1, mean_d2 = np.mean(results_d1), np.mean(results_d2)
        std_d1, std_d2 = np.round(std_err(results_d1), 2), np.round(std_err(results_d2), 2)

        fig, (ax0, ax1) = plt.subplots(1, 2, num=str(sam_meas_.sample.name) + "_single_sweeps")
        ax0.plot(sweeps, results_d1, label="Dicke erste Schicht")
        ax0.axhline(mean_d1, label=f"Durchschnittliche Dicke ({mean_d1}$\pm${std_d1} $\mu$m)", c="red")
        ax0.axhline(truth_d1, label=f"Messschieber Messung", c="blue")

        ax1.plot(sweeps, results_d2, label="Dicke zweite Schicht")
        ax1.axhline(mean_d2, label=f"Durschnittliche Dicke ({mean_d2}$\pm${std_d2} $\mu$m)", c="red")
        ax1.axhline(truth_d2, label=f"Messschieber Messung", c="blue")

        ax0.set_xlabel("Aufnahme Index")
        ax0.set_ylabel("Dicke bester Fit ($\mu$m)")
        ax1.set_ylabel("Dicke bester Fit ($\mu$m)")

        fig, (ax0, ax1) = plt.subplots(2, 1, num=str(sam_meas_.sample.name) + "_single_sweeps_losses")
        ax0.plot(sweeps, min_losses)
        ax1.plot(sweeps, results_d1, label="Dicke erste Schicht")
        ax1.plot(sweeps, results_d2, label="Dicke zweite Schicht")
        ax0.set_xlabel("Aufnahme Index")
        ax0.set_ylabel("Kleinstes Residuum")
        ax1.set_ylabel("Dicke bester Fit ($\mu$m)")

    def single_layer_eval(self, sam_meas_: Measurement, ts_meas_: Measurement, mod_meas_: ModelMeasurement):
        selected_system = self.__options["selected_system"]

        ts_f_idx = []
        for selected_freq in sam_meas_.freq:
            ts_f_idx.append(np.argmin(np.abs(selected_freq - ts_meas_.freq)))

        r_amp_ts, r_phi_ts = np.abs(ts_meas_.r_avg[ts_f_idx]), np.angle(ts_meas_.r_avg[ts_f_idx])
        r_amp_truth, r_phi_truth = np.abs(mod_meas_.r_avg[ts_f_idx]), np.angle(mod_meas_.r_avg[ts_f_idx])
        r_real_truth, r_imag_truth = mod_meas_.r_avg[ts_f_idx].real, mod_meas_.r_avg[ts_f_idx].imag

        d_truth = sam_meas_.sample.value.thicknesses

        cost = Cost(sam_meas_)

        avg_cost = cost.calc_cost

        avg_cost(d_truth, verbose=True)

        d_min, d_max = np.max([1, d_truth[0] - 150]), d_truth[0] + 150
        d1 = np.arange(int(d_min), int(d_max), 1, dtype=float)
        losses = []
        for d in d1:
            p = np.array([d])
            err = avg_cost(p)
            losses.append(err)

        plt.figure(str(sam_meas_.sample.name) + "_avg")
        plt.plot(d1, losses, label=sam_meas_, c=self.plot_vars["colors"][sam_meas_.system.value])
        if not self.plot_vars["truth_line_exists"]:
            plt.axvline(x=sam_meas_.sample.value.tot_thickness, label="True thickness", c="red")
            self.plot_vars["truth_line_exists"] = True
        plt.axvline(x=d1[np.argmin(losses)],
                    label=f"Gefunden {d1[np.argmin(losses)]} ($\mu$m) ({sam_meas_.system.name})",
                    linestyle="--", c=self.plot_vars["colors"][sam_meas_.system.value])
        plt.xlabel("Dicke erste Shicht ($\mu$m)")
        plt.ylabel("Summe der Residuen")
        print(f"Found minimum: {d1[np.argmin(losses)]} (all sweeps averaged)")

        if sam_meas_.system != selected_system:
            return

        n_sweeps = sam_meas_.n_sweeps
        sweeps = list(range(n_sweeps))
        results = np.zeros(n_sweeps, dtype=float)
        min_losses = np.zeros(n_sweeps)
        for sweep_idx in sweeps:
            sweep_cost = partial(cost.calc_cost, sweep_idx_=sweep_idx)
            losses = []
            for d in d1:
                p = np.array([d])
                err = sweep_cost(p)
                losses.append(err)
            best_fit = d1[np.argmin(losses)]
            best_fit_loss = np.min(losses)
            print(f"Found minimum: {best_fit} ({best_fit_loss}), sweep: {sweep_idx}")
            min_losses[sweep_idx] = best_fit_loss
            results[sweep_idx] = best_fit

        mean = np.mean(results)
        std = np.round(std_err(results), 2)

        plt.figure(str(sam_meas_.sample.name) + "_single_sweeps")
        plt.scatter(sweeps, results, marker="x", s=15)
        plt.axhline(mean, label=f"Durchschnittliche Dicke ({mean}$\pm${std} um)", c="red")
        plt.axhline(d_truth, label=f"Messschieber Messung", c="blue")
        plt.xlabel("Aufnahme Index")
        plt.ylabel("Dicke bester Fit ($\mu$m)")

        fig, (ax0, ax1) = plt.subplots(2, 1, num=str(sam_meas_.sample.name) + "_single_sweeps_losses")
        ax0.plot(sweeps, min_losses)
        ax1.plot(sweeps, results)
        ax0.set_xlabel("Aufnahme Index")
        ax0.set_ylabel("Kleinstes Residuum")
        ax1.set_ylabel("Dicke bester Fit ($\mu$m)")

    def integrated_std_err_plot(self, std_err_=None):
        selected_sample_ = self.__options["selected_sample"]
        dt = 0.960  # ms
        n_sweeps_ = 2000
        layer_cnt = selected_sample_.value.layers
        std_err_dict = {SamplesEnum.ampelMannLeft: [0.02, 0.02, 0.06],
                    SamplesEnum.ampelMannRight: [0.12, 0.10, 0.07]}
        if std_err_ is None:
            try:
                std_err_ = np.array(std_err_dict[selected_sample_], dtype=float)
            except KeyError:
                std_err_ = np.zeros(layer_cnt)

        sigma = std_err_ * np.sqrt(n_sweeps_)
        n = np.arange(1, 20 + 1, 1)

        plt.figure(f"Integrated_standard_error_{selected_sample_.name}")
        plt.title("Standardfehler in abhängigkeit von der Messzeit")
        plt.xlabel("Messzeit (ms)")
        plt.ylabel("Standardfehler ($\mu$m)")

        for layer_idx in range(layer_cnt):
            std_err_ = sigma[layer_idx] / np.sqrt(n)
            label = f"Schicht {layer_idx + 1} ($\sigma =${np.round(sigma[layer_idx], 2)} $\mu$m)"
            plt.plot(n * dt, std_err_, label=label, lw=3.0, markersize=5)


if __name__ == '__main__':
    save_plots = False

    # selected_sweep: int, None or "random" # 52 b ## 593 g # 1319 b # 519 b # 420 used in report /w PIC
    options = {"selected_system": SystemEnum.WaveSource, "selected_sample": SamplesEnum.ampelMannLeft,
               "selected_sweep": 12,
               "less_plots": True,
               "single_sweep_eval": True,
               "debug_info": False,
               "plot_grid": False
               }

    save_dir = result_dir / "JumpingLaser" / str(options["selected_system"].name) / str(options["selected_sample"].name)
    save_dir.mkdir(parents=True, exist_ok=True)
    new_rcparams = {"savefig.directory": save_dir}
    mpl.rcParams = mpl_style_params(new_rcparams)

    options["save_dir"] = save_dir
    new_eval = JumpingLaserEval(options)

    new_eval.calc_sample_refl_coe()
    new_eval.plot_sample_refl_coe()
    new_eval.thickness_eval()

    # integrated_std_err_plot()

    plt_show(mpl, en_save=save_plots)
