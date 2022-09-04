import numpy as np
from consts import THz, ones, array


def get_n(freqs, n_min=2.71, n_max=2.86):
    freqs_full = np.arange(0.300, 1.500 + 0.001, 0.001) * THz
    m = len(freqs_full)
    a = (n_max - n_min) / m

    n1 = np.arange(0, m)*a + n_min

    n = np.array([ones(m), 1.50*ones(m), n1, 1.50*ones(m), ones(m)], dtype=np.complex128).transpose()

    selected_freqs_idx = array([np.argwhere(np.isclose(freq, freqs_full))[0][0] for freq in freqs])

    return n[selected_freqs_idx,:]
