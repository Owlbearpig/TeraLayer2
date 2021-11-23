import numpy as np
from numpy import arange, array
from functions import residuals
import matplotlib.pyplot as plt
from multir_numba import multir_numba
from functions import format_data
from consts import default_mask, um, d_best

"""
Before going to C check if algo is able to optimize multir in a reasonable time
"""

"""
def fun(x, p=(5, 5, 5)):
    return p[0] + p[1]*x + p[2]*x**2
    
x = arange(0, 10, 1)
# model with noise
p0 = (5.5, 4.6, 4.901)
y0 = fun(x, p0) + 1 * np.random.random(x.shape)

lb = [4, 4, 4]
ub = [6, 6, 6]
"""

lam, R = format_data()

fun = multir_numba
x = lam[default_mask]
y0 = R[default_mask]

print(sum(residuals(d_best, fun, x, y0)))

lb = array([0.000001, 0.00001, 0.000001])
ub = array([0.001, 0.001, 0.001])

max_its = 10
rez = 50

grd_x0, grd_x1 = lb[0], ub[0]
grd_y0, grd_y1 = lb[1], ub[1]
grd_z0, grd_z1 = lb[2], ub[2]

# initial 'full' grid matching bounds
grd_x = np.linspace(grd_x0, grd_x1, rez)
grd_y = np.linspace(grd_y0, grd_y1, rez)
grd_z = np.linspace(grd_z0, grd_z1, rez)

min_err = 1e100
vals = 0  # count points searched
x_idx_best, y_idx_best, z_idx_best = None, None, None
ps = np.zeros((max_its, 3))

its = 0
while its < max_its:
    for i in range(0, rez):
        print(f'{i} / {rez}')
        for j in range(0, rez):
            for k in range(0, rez):
                p_new = array([grd_x[i], grd_y[j], grd_z[k]])
                y_new = fun(x, p_new)
                err = sum((y0 - y_new) ** 2)
                vals = vals + 1

                if err < min_err:
                    x_idx_best, y_idx_best, z_idx_best = i, j, k
                    min_err = err
    print(x_idx_best, y_idx_best, z_idx_best)
    # refine grid to around best vals

    # step size is constant and equal in all directions
    step_size = 10*(grd_x[1] - grd_x[0])
    print()
    print('best d (=p):', grd_x[x_idx_best]*um, grd_y[y_idx_best]*um, grd_z[z_idx_best]*um)
    print('residual sum (min_err): ', min_err)
    print('step size:', step_size)
    print('x range:', min(grd_x), max(grd_x))
    print('y range:', min(grd_y), max(grd_y))
    print('z range:', min(grd_z), max(grd_z))
    print()
    grd_x = np.linspace(grd_x[x_idx_best]-step_size, grd_x[x_idx_best]+step_size, rez)
    grd_y = np.linspace(grd_y[y_idx_best]-step_size, grd_y[y_idx_best]+step_size, rez)
    grd_z = np.linspace(grd_z[z_idx_best]-step_size, grd_z[z_idx_best]+step_size, rez)

    # save best result at each iteration
    p = [grd_x[x_idx_best], grd_y[y_idx_best], grd_z[z_idx_best]]
    ps[its] = p

    its = its + 1

res_sums = []
for p in ps:
    res_sum = sum(residuals(p, fun, x, y0))
    res_sums.append(res_sum)

plt.plot(arange(0, max_its), res_sums)
plt.show()


plt.plot(lam/1e-3, R, label='measurement')
plt.plot(lam[default_mask]/1e-3, R[default_mask], 'o', color='red')
plt.plot(lam/1e-3, fun(lam, ps[-1]), label='fit')
plt.plot(lam/1e-3, fun(lam, d_best), label='best fit (scipy/matlab LM-algo)')
plt.xlim((0, 2))
plt.ylim((0, 1.1))
plt.xlabel('THZ-Wavelenght (mm)')
plt.ylabel('$r^2$ (arb. units)')
plt.legend()
plt.show()

plt.plot(x/1e-3, y0, label='measurement')
plt.plot(x/1e-3, fun(x, ps[-1]), label='fit')
plt.plot(x/1e-3, fun(x, d_best), label='best fit (scipy/matlab LM-algo)')
plt.xlabel('THZ-Wavelenght (mm)')
plt.ylabel('$r^2$ (arb. units)')
plt.legend()
plt.show()
