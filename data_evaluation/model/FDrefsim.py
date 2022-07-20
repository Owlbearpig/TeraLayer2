import numpy as np
import matplotlib.pyplot as plt


def do_fft(t, y):
    n = len(y)
    dt = np.float(np.mean(np.diff(t)))
    Y = np.fft.fft(y, n)
    f = np.fft.fftfreq(len(t), dt)
    idx_range = f > 0

    return f[idx_range], Y[idx_range]

def pulse(t):
    A, B, C = 0.15, 10, 0.3

    pre_fac = (A*(B-t)**2 - A*C**2)/C**4

    y = pre_fac*np.exp(-(t-B)**2 / (2*C**2))

    y += np.random.random(len(y)) * 0.2 * np.sin(2*np.pi*np.linspace(0, 100, len(y)))


    return y

fig, (ax1, ax2) = plt.subplots(2, 1)

t = np.linspace(0,30,2000)
y = pulse(t)

f, Y = do_fft(t, y)
f_slice = f < 4.5

ax1.plot(t, y)
ax2.plot(f[f_slice], 20*np.log10(np.abs(Y)[f_slice]))

f_slice1 = f > 1
y = np.fft.ifft(Y[f_slice1])

plt.figure()
plt.plot(y)

Y = np.fft.fft(y)

plt.figure()
plt.plot(20*np.log10(np.abs(Y)))


plt.show()


