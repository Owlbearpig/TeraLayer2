from parse_data import get_all_measurements, SystemEnum, MeasTypeEnum
import numpy as np

all_measurements = get_all_measurements()
blue_cube_measurements = [meas for meas in all_measurements if meas.sample.s_id == 1]
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
        meas.r_exp_car = meas.data_fd_car / ref_meas.data_fd_car
        print(meas.r_exp_car.shape)


if __name__ == '__main__':
    calc_reflection_coe(blue_cube_measurements)
