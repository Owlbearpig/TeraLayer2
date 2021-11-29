from results import d_best, p_full_search
from plotting import plot_result
from functions import calc_full_loss

plot_result(d_best)
plot_result(p_full_search)

print('d_best: ', calc_full_loss(d_best))
print('p_full_search: ', calc_full_loss(p_full_search))
