import numpy as np
import matplotlib.pyplot as plt

ts = 8000
n = np.arange(0, 8)
t = n*(1/ts)
xn = np.sin(2*np.pi*1000*t) + 0.5*np.sin(2*np.pi*2000*t + 3*np.pi/4)

plt.plot(n, xn)
plt.show()
