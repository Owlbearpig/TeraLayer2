import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft

def box_smoothing(data):
    box = np.ones(50)/50
    data_smooth = np.convolve(data-np.mean(data[0:50]), box, mode = "same")
    data_smooth += np.mean(data[0:50])
    #print(np.mean(data[0:50]))
    return data_smooth

x = np.linspace(0, 4*np.pi, 1000)
data = np.sin(x)
data += 0.01*np.random.random(len(data))
smooth_data = box_smoothing(data)

plt.plot(smooth_data, label="smooth_data")
plt.plot(data, label="data")
plt.legend()
plt.show()

fs = fft(data)
fs_smooth = fft(smooth_data)

plt.plot(np.abs(fs_smooth), label="smooth_data")
plt.plot(np.abs(fs), label="data")
plt.yscale('log')
plt.legend()
plt.show()
