from parse_data import get_all_measurements, SystemEnum, MeasTypeEnum
import numpy as np
import matplotlib.pyplot as plt

all_measurements = get_all_measurements()

ref_measurements = [meas for meas in all_measurements if meas.meas_type == MeasTypeEnum.Reference]


def find_nearest_ref(meas):
    ref_same_system = [ref for ref in ref_measurements if ref.system == meas.system]
    abs_time_diffs = []
    for ref_meas in ref_same_system:
        abs_time_diff = np.abs(ref_meas.time_diff(meas))
        abs_time_diffs.append((abs_time_diff, ref_meas))

    sorted_refs = sorted(abs_time_diffs, key=lambda x: x[0])

    return sorted_refs[0][1]


def calc_reflection_coe(measurements):
    for meas in measurements:
        ref_meas = find_nearest_ref(meas)
        meas.r_exp_car = meas.data_car / ref_meas.data_car
        if meas.system == SystemEnum.TSweeper:
            plt.figure("amp")
            plt.plot(meas.freq, np.log10(np.abs(meas.r_exp_car)))

            plt.figure("phase")
            plt.plot(meas.freq, np.angle(meas.r_exp_car))


if __name__ == '__main__':
    samples = [meas for meas in all_measurements if meas.sample.s_id == 3]
    print(samples)

    calc_reflection_coe(samples)

    plt.show()
