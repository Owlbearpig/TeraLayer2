import numpy as np
from numpy import arange
from functions import residuals
import matplotlib.pyplot as plt


def fun(x, p=(5, 5)):
    return p[0] + p[1]*x


x = arange(0, 10, 1)
# model with noise
y0 = fun(x, (5, 5)) + 0.2 * np.random.random(x.shape)

lb = [4, 4]
ub = [6, 6]
max_its = 4
rez = 100

its = 0
a_m = lb[0]
a_M = ub[0]
b_m = lb[1]
b_M = ub[1]

as_ = (a_M - a_m) * arange(0, rez - 1) / (rez - 1) + a_m
bs_ = (b_M - b_m) * arange(0, rez - 1) / (rez - 1) + b_m

min_err = 1e100
vals = 0  # count points searched
i_a, i_b = 0, 0  # store argument(index in this case) of best result
ps = np.zeros((max_its, 2))
while its < max_its:  # this has sligtly differnt sintax in C
    # Grid search 1
    for j in range(0, rez - 1):  # this has sligtly differnt sintax in C
        for k in range(0, rez - 1):  # this has sligtly differnt sintax in C
            yy = as_[j] + bs_[k] * x  # this line will need extra C programming
            err = sum((y0 - yy) ** 2)  # this line will need extra C programming
            vals = vals + 1

            if err < min_err:
                i_a = j
                i_b = k
                min_err = err
    # Grid search 1
    print(i_a, i_b)
    # make refined grid axes
    if i_a == 0:  # boundary case left
        as_ = (as_[i_a + 1] - as_[i_a]) * arange(0, rez - 1) / (rez - 1) + as_[i_a]
    else:
        if i_a == rez:  # boundary case right
            as_ = (as_[i_a] - as_[i_a - 1]) * arange(0, rez - 1) / (rez - 1) + as_[i_a - 1]
        else:  # somewhere not on edge of old grid
            as_ = (as_[i_a + 1] - as_[i_a - 1]) * arange(0, rez - 1) / (rez - 1) + as_[i_a - 1]
    if i_b == 0:  # boundary case left
        bs_ = (bs_[i_b + 1] - bs_[i_b]) * arange(0, rez - 1) / (rez - 1) + bs_[i_b]
    else:
        if i_b == rez:  # boundary case right
            bs_ = (bs_[i_b] - bs_[i_b - 1]) * arange(0, rez - 1) / (rez - 1) + bs_[i_b - 1]
        else:  # somewhere not on edge of old grid
            bs_ = (bs_[i_b + 1] - bs_[i_b - 1]) * arange(0, rez - 1) / (rez - 1) + bs_[i_b - 1]

    #TODO question: Shouldn't new grid be made after storing minimum? otherwise we use refined grid

    # save best result for each iteration
    p = [as_[i_a], bs_[i_b]]
    ps[its] = p

    its = its + 1

res_sums = []
for p in ps:
    res_sum = sum(residuals(p, fun, x, y0))
    res_sums.append(res_sum)

plt.plot(arange(0, max_its), res_sums)
plt.show()


plt.plot(x, y0, label='measurement')
plt.plot(x, p[0] + p[1] * x, label='fit')
plt.legend()
plt.show()
