import tmm
from numpy import array
import numpy as np
from scipy.constants import c
from consts import um_to_m, GHz
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
from scipy.constants import c as c0
from tmm_package import coh_tmm_slim

freq = 0.75
lam = c0 * 1e-6 / freq
print(f"Frequency: {freq} THz, wavelength {lam} um")
n1, n2 = 2.9, 2.9
d_list = [np.inf, 45.0, 90.0, np.inf]
r = coh_tmm_slim("s", [1, n1, n2, 1], d_list, 0, lam)
r_abs, r_ang = np.abs(r), np.angle(r)

r0, r1, r2 = (1-n1)/(1+n1), (n1-n2)/(n1+n2), (n2-1)/(n2+1)
# phi = (2 * np.random.random() - 1) * np.pi
eu = np.exp(1j * r_ang)

d1, d2 = np.arange(1, 155, 1), np.arange(1, 155, 1)


def f(d1_, d2_):
    phi1, phi2 = 2*np.pi*d1_*n1/lam, 2*np.pi*d2_*n2/lam
    x_, y_ = np.exp(1j * 2 * phi1), np.exp(1j * 2 * phi2)
    c0_ = r2 - r_abs * r0 * r2 * eu
    c1_ = r0 * r1 * r2 - r_abs * r1 * r2 * eu
    c2_ = r1 - r_abs * r0 * r1 * eu
    c3_ = r0 - r_abs * eu

    return c0_ + c1_ * x_ + c2_ * y_ + c3_ * x_ * y_


fig0, ax0 = plt.subplots(subplot_kw={"projection": "3d"})
fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
# Make data.
X, Y = np.meshgrid(d1, d2)
Z = f(X, Y)

# Plot the surface.
ax0.set_title(f"Real part ({freq} THz)")
surf0 = ax0.plot_surface(X, Y, Z.real, cmap=cm.viridis,
                        linewidth=0, antialiased=False)
ax0.scatter(*d_list[1:3], f(*d_list[1:3]).real, s=75, color='red')
print(f(*d_list[1:3]))
ax1.set_title(f"Imag part ({freq} THz)")
surf1 = ax1.plot_surface(X, Y, Z.imag, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

print(np.isclose(Z.real, 0))
print(*np.where(np.isclose(Z.real, 0)))

plt.figure()
for i in range(5):
    # y = f(d1, d_list[2]+i)
    y = array([f(k, l+i) for (k, l) in zip(reversed(d1), d2)])
    plt.plot(d1, y.real, label=d_list[1]+i)
    plt.scatter(d1, y.imag, label=d_list[1] + i)
plt.title(f"{freq} THz")
plt.axhline(y=0.0, color='r', linestyle='-')
plt.legend()

# Customize the z axis.
ax0.set_zlim(-1.01, 1.01)
ax0.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax0.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig0.colorbar(surf0, shrink=0.5, aspect=5)
fig1.colorbar(surf1, shrink=0.5, aspect=5)
plt.show()
