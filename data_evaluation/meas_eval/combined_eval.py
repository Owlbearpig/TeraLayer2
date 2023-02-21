import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from functions import do_fft
from tds.main import load_data as load_tds_data
from cw.main import load_data as load_cw_data

sam_idx = 35

ref_cw_fd, sam_cw_fd = load_cw_data(sam_idx=sam_idx, shift=0)
r_cw_fd = array([ref_cw_fd[:, 0].real, sam_cw_fd[:, 1] / ref_cw_fd[:, 1]]).T

ref_tds_td, sam_tds_td = load_tds_data(sam_idx=sam_idx, signal_shift=0)
sam_tds_fd, ref_tds_fd = do_fft(sam_tds_td), do_fft(ref_tds_td)
r_tds_fd = array([ref_tds_fd[:, 0].real, sam_tds_fd[:, 1] / ref_tds_fd[:, 1]]).T

plt.figure("Phase")
# plt.plot(sam_fd[:, 0], np.angle(sam_fd[:, 1]), label="$\phi_{sam}$")
# plt.plot(sam_fd[:, 0], np.angle(ref_fd[:, 1]), label="$\phi_{ref}$")
plt.plot(r_tds_fd[:, 0].real, np.angle(r_tds_fd[:, 1]), label=r"TDS $\phi_{sam} - \phi_{ref}$ " + f"{sam_idx}")
plt.plot(r_cw_fd[:, 0].real, np.roll(np.angle(r_cw_fd[:, 1]), 0),
         label=r"CW $\phi_{sam} - \phi_{ref}$ " + f"{sam_idx}")
plt.xlim((-0.1, 1.5))
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (Rad)")
plt.legend()

plt.figure("Spectrum")
plt.plot(r_tds_fd[:, 0], 20 * np.log10(np.abs(r_tds_fd[:, 1])), label=f"TDS {sam_idx}")
plt.plot(r_cw_fd[:, 0], 20 * np.log10(np.abs(r_cw_fd[:, 1])), label=f"CW {sam_idx}")
plt.xlim((-0.1, 1.5))
plt.xlabel("Frequency (THz)")
plt.ylabel("Amplitude (dB)")
plt.legend()

plt.show()
