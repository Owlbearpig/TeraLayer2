import matplotlib.pyplot as plt
import numpy as np

signal = np.zeros(1000) # + np.random.random(1000)
signal[198] = 3
signal[199] = 7
signal[200] = 10
signal[210] = 10
signal[211] = 7
signal[212] = 3
signal[798] = -1
signal[799] = -4
signal[800] = -5
signal[810] = -5
signal[811] = -4
signal[812] = -1

fft_signal = np.fft.fft(signal)

plt.figure('fft signal')
plt.plot(np.abs(fft_signal))
plt.yscale('log')
plt.show()

plt.figure('fft phase')
plt.plot(np.unwrap(np.arctan2(fft_signal.imag, fft_signal.real)))
plt.show()

signal_ifft = np.fft.ifft(fft_signal)
print(np.mean(signal_ifft.imag))
#plt.plot(signal_ifft.real)
plt.plot(signal, label='truth')
plt.legend()
plt.show()

