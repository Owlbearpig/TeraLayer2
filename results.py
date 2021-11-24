from functions import plot
import numpy as np
from numpy import array
from consts import um_to_m

d_best = np.array([37.29533693, 626.64077655, 37.2953365])*um_to_m # from scipy.optimize (least_sq)
p_brutef = array([205.2044088176353, 638.9178356713428, 797.797595190381])*um_to_m  # d2 > 100 um
p_brutef_g = array([57.05611222444891, 39.75951903807616, 605.6052104208418])*um_to_m  # global minimum (2 um rez)
p_test = array([37.05611222444891, 626.6052104208418, 39.75951903807616])*um_to_m

if __name__ == '__main__':
    #plot(p_brutef_g)
    plot(d_best)
