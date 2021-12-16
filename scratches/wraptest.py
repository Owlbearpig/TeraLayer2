import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
# since sin / cos are periodic with period 2pi it should be possible to map any number to the range -pi, pi I think
"""
sin = np.sin(np.linspace(-2*pi, 30, 1000))
x = 30*np.random.random(100)
print(x)
y = x % (2*pi) - pi

plt.plot(np.linspace(-2*pi, 30, 1000), sin)
plt.scatter(x, np.zeros(x.shape))
plt.scatter(y, np.zeros(x.shape))
plt.vlines(-pi, -1, 1)
plt.vlines(pi, -1, 1)
plt.show()
"""
print(2*pi)
x = -30*np.random.rand()
print(x)

print(x % (2*pi))
print(x - int(x/(2*pi))*2*pi)

