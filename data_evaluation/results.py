from functions import calc_loss, calc_full_loss
from visualizing.plotting import plot_result
from consts import *

d_best = np.array([37.29533693, 626.64077655, 37.2953365])*um_to_m # from scipy.optimize (least_sq)
p_brutef = array([205.2044088176353, 638.9178356713428, 797.797595190381])*um_to_m  # d2 > 100 um
p_full_search = array([57.05611222444891, 39.75951903807616, 605.6052104208418])*um_to_m  # global minimum (2 um rez)
p_test = array([37.05611222444891, 626.6052104208418, 39.75951903807616])*um_to_m
p_high_res = array([51, 619.5979899497487, 39])*um_to_m
p_C_simplex_sample10 = array([44.07540263, 630.24273320, 44.07565431])*um_to_m

if __name__ == '__main__':
    mask = custom_mask_420
    sample_idx = 10
    print(calc_loss(d_best, mask=mask, sample_file_idx=sample_idx))
    print(calc_loss(p_C_simplex_sample10, mask=mask, sample_file_idx=sample_idx), '\n')
    print(calc_full_loss(d_best, sample_file_idx=sample_idx))
    print(calc_full_loss(p_C_simplex_sample10, sample_file_idx=sample_idx))
    #plot(p_brutef_g)
    plot_result(p_C_simplex_sample10, mask=mask, sample_file_idx=sample_idx)
