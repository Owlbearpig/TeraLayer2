from parse_data import get_all_measurements, SystemEnum, MeasTypeEnum
import numpy as np
import matplotlib.pyplot as plt
from helpers import plt_show

all_measurements = get_all_measurements()

ref_measurements = [meas for meas in all_measurements if meas.meas_type == MeasTypeEnum.Reference]


def find_nearest_ref(meas):
    ref_same_system = [ref for ref in ref_measurements if ref.system == meas.system]
    abs_time_diffs = []
    for ref_meas in ref_same_system:
        abs_time_diff = np.abs(ref_meas.time_diff(meas))
        abs_time_diffs.append((abs_time_diff, ref_meas))

    sorted_refs = sorted(abs_time_diffs, key=lambda x: x[0])
    closest_ref = sorted_refs[0][1]

    print(meas, closest_ref)

    return closest_ref


def calc_reflection_coe(measurements):
    for meas in measurements:
        ref_meas = find_nearest_ref(meas)
        meas.r_exp_car_avg = meas.data_car_avg / ref_meas.data_car_avg

        if meas.system == SystemEnum.TSweeper:
            plt.figure("TSWeeper amp")
            plt.title("TSWeeper amp")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")
            plt.xlim((-0.150, 2.1))

            plt.plot(ref_meas.freq, np.log10(np.abs(ref_meas.amp_avg)), label="reference")
            plt.plot(meas.freq, np.log10(np.abs(meas.amp_avg)), label="sample")

            plt.figure("TSWeeper phase")
            plt.title("TSWeeper phase")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Phase")
            plt.xlim((-0.150, 2.1))

            plt.plot(ref_meas.freq, ref_meas.phase_avg, label="reference")
            plt.plot(meas.freq, meas.phase_avg, label="sample")

        plt.figure("r_exp amp")
        plt.title(meas.sample)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")
        plt.xlim((-0.150, 2.1))

        if meas.system == SystemEnum.TSweeper:
            plt.plot(meas.freq, np.log10(np.abs(meas.r_exp_car_avg)), label=meas.system, c="r")
        else:
            plt.scatter(meas.freq, np.log10(np.abs(meas.r_exp_car_avg)), label=meas.system, s=22, zorder=9)

        plt.figure("r_exp phase")
        plt.title(meas.sample)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")
        plt.xlim((-0.150, 2.1))

        if meas.system == SystemEnum.TSweeper:
            plt.plot(meas.freq, np.angle(meas.r_exp_car_avg), label=meas.system, c="r")
        else:
            plt.scatter(meas.freq, np.angle(meas.r_exp_car_avg), label=meas.system, s=22, zorder=9)


if __name__ == '__main__':
    samples = [meas for meas in all_measurements if meas.sample.s_id == 17]

    calc_reflection_coe(samples)

    plt_show(plt)
