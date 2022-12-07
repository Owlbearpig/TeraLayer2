import matplotlib.pyplot as plt
from consts import custom_mask_420, um_to_m, THz, GHz, um, pi
import numpy as np
from numpy import array, sum
import matplotlib as mpl
from model.cost_function import Cost

plt.rcParams['figure.constrained_layout.use'] = True

p_sol = array([211., 660.,  64.], dtype=float)

freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950]) * THz # GHz; freqs. set on fpga
freqs_all = np.arange(0.0, 1400.0, 1) * GHz

new_cost = Cost(freqs_all, p_sol, 0.00)
new_cost_noisy = Cost(freqs_all, p_sol, 0.50)

r0_amp = new_cost.R0_amplitude.flatten()
r0_phase = new_cost.R0_phase.flatten()
r0_amp_noisy = new_cost_noisy.R0_amplitude.flatten()
r0_phase_noisy = new_cost_noisy.R0_phase.flatten()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.vlines(freqs/GHz, ymin=-pi, ymax=pi, color="red", label="Selected frequencies")
ax1.plot(freqs_all / GHz, r0_phase_noisy, label="Phase noisy")
ax1.plot(freqs_all / GHz, r0_phase, label="Phase")
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("Phase (rad)")
ax1.legend()

ax2.vlines(freqs/GHz, ymin=0, ymax=0.8, color="red", label="Selected frequencies")
ax2.plot(freqs_all / GHz, r0_amp_noisy, label="Intensity noisy")
ax2.plot(freqs_all / GHz, r0_amp, label="Intensity")

ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Reflectance")
ax2.legend()
fig.suptitle("Example model data p_sol = [211., 660.,  64.]")

plt.show()