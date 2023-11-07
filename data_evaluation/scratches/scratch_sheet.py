from numpy import array
import numpy as np
from scipy.constants import c
from consts import um_to_m, GHz
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
from scipy.constants import c as c0

lam = c0 * 1e-6  # 1 THz wavelength

r, r0, r1, r2 = 0.05, 0.4, 0.4, 0.6
phi = (2 * np.random.random() - 1) * np.pi
eu = np.exp(1j * 2 * phi)

d1, d2 = np.linspace(1, 100, 1000), np.linspace(1, 100, 1000)


def f(d1_, d2_):
    phi1, phi2 = 2*np.pi*d1_*1.5/lam, 2*np.pi*d2_*2.9/lam
    x_, y_ = np.exp(1j * 2 * phi1), np.exp(1j * 2 * phi2)
    c0_ = r2 - r * r0 * r2 * eu
    c1_ = r0 * r1 * r2 - r * r1 * r2 * eu
    c2_ = r1 - r * r0 * r1 * eu
    c3_ = r0 - r * eu

    return c0_ + c1_ * x_ + c2_ * y_ + c3_ * x_ * y_


fig0, ax0 = plt.subplots(subplot_kw={"projection": "3d"})
fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
# Make data.
X, Y = np.meshgrid(d1, d2)
Z = f(X, Y)

# Plot the surface.
surf0 = ax0.plot_surface(X, Y, Z.real, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
surf1 = ax1.plot_surface(X, Y, Z.imag, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
# Customize the z axis.
ax0.set_zlim(-1.01, 1.01)
ax0.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax0.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig0.colorbar(surf0, shrink=0.5, aspect=5)
fig1.colorbar(surf1, shrink=0.5, aspect=5)
plt.show()
