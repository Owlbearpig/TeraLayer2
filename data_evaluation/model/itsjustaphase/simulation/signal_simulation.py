# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:00:13 2021

@author: Talebf
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm


#######################################
def thz_pulse(t, tau=0.15, loc=0):
    return t * norm.pdf(t, scale=tau)


def thz_pulse2(t, tau=0.15):
    return (t / tau ** 2) * (np.exp(-t ** 2 / tau ** 2))


# tau = 0.3  # mm/ps
tau = 0.4  # mm/ps
#dt = 0.33  # Ps
dt = 0.33  # Ps
# dt = 0.01  # Ps
t = np.arange(0, 1000, dt)
t_r = 46.45
t_0 = t_r + 0
n0 = 1
d1 = 0.040
n1 = 1.50
d2 = 0.640
n2 = 2.80
d3 = 0.075
n3 = 1.50

t_1 = 2 * (d1 * n1) / 0.3 + t_0
ab1 = np.exp(-0.5)
t_2 = 2 * (d2 * n2) / 0.3 + t_1
ab2 = np.exp(-0.5)
t_3 = 2 * (d3 * n3) / 0.3 + t_2
ab3 = np.exp(-0.5)

print("1-layer:", t_1 - t_0, "2-layer:", t_2 - t_1, "3-layer:", t_3 - t_2)
r0 = (n0 - n1) / (n0 + n1)
t01 = 2 * n0 / (n0 + n1)
t10 = 2 * n1 / (n0 + n1)
r1 = (n1 - n2) / (n1 + n2)
t12 = 2 * n1 / (n1 + n2)
t21 = 2 * n2 / (n1 + n2)
r2 = (n2 - n3) / (n2 + n3)
t23 = 2 * n2 / (n2 + n3)
t32 = 2 * n3 / (n2 + n3)

r3 = (n3 - n0) / (n3 + n0)
r4 = -1

y = r4 * thz_pulse2(t - t_r, tau)  # Referenz
y2 = r0 * thz_pulse2(t - t_0, tau) + \
     ab1 * t01 * r1 * t10 * thz_pulse2(t - t_1, tau) + \
     ab1 * ab2 * t01 * t12 * r2 * t10 * t21 * thz_pulse2(t - t_2, tau) + \
     ab1 * ab2 * ab3 * t01 * t12 * t23 * r3 * t32 * t21 * t10 * thz_pulse2(t - t_3, tau)

# Noise
# y  += np.random.random(len(t)) * np.max(np.abs(y))
#y2 += np.random.random(len(t)) * np.max(np.abs(y2)) * 0.02
#y += np.random.random(len(t)) * np.max(np.abs(y)) * 0.02

Y = np.fft.fft(y)
Y2 = np.fft.fft(y2)
freq = np.fft.fftfreq(len(t), dt)
idx = freq > 0

# np.log10 = lambda x: x / 20
plt.figure("Amplitude frequency domain (Sim)")
plt.plot(freq[idx], 20 * np.log10(np.abs(Y[idx])), label="Reference")
plt.plot(freq[idx], 20 * np.log10(np.abs(Y2[idx])), label="Sample")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")

np.unwrap = lambda x: x
plt.figure("Phase frequency domain (Sim)")
plt.plot(freq[idx], np.unwrap(np.angle(Y[idx])), label="Reference")
plt.plot(freq[idx], np.unwrap(np.angle(Y2[idx])), label="Sample")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")

y = np.fft.ifft(Y)
y2 = np.fft.ifft(Y2)

plt.figure("Amplitude time domain (Sim)")
plt.plot(t, y, label="Reference")
plt.plot(t, y2, label="Sample")
plt.legend()
plt.show()

dx = 1
dy = 1
tilt = 8 / 180 * np.pi

k = 0
for i in range(1):
    for j in range(1):
        k += 1
        t_2 = 2 * (d2 * n2) / 0.3 + t_1 + np.tan(tilt) * i * dx / 0.3
        y2 = r0 * thz_pulse2(t - t_0, tau) + \
             ab1 * t01 * r1 * t10 * thz_pulse2(t - t_1, tau) + \
             ab1 * ab2 * t01 * t12 * r2 * t10 * t21 * thz_pulse2(t - t_2, tau) + \
             ab1 * ab2 * ab3 * t01 * t12 * t23 * r3 * t32 * t21 * t10 * thz_pulse2(t - t_3, tau)
        y2 += np.random.random(len(t)) * np.max(np.abs(y2)) * 0.02
        temp = np.vstack((t, y2)).T
