from numpy import array
from results import d_best, p_full_search
from plotting import plot_result
from functions import calc_full_loss
from consts import um_to_m

d_custom = array([44, 680, 36])*um_to_m

plot_result(d_custom)
plot_result(d_best)
plot_result(p_full_search)

print('d_best: ', calc_full_loss(d_best))
print('p_full_search: ', calc_full_loss(p_full_search))
