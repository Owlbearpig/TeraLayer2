import numpy as np
from consts import THz, ones, array


def get_n(freqs, n_min=2.71, n_max=2.86):
    freqs_full = np.arange(0.000, 1.500 + 0.001, 0.001) * THz
    m = len(freqs_full)
    a = (n_max - n_min) / m

    n1 = np.arange(0, m)*a + n_min

    #n = np.array([ones(m), 1.50*ones(m), n1, 1.50*ones(m), ones(m)], dtype=np.complex128).transpose()
    n = np.array([ones(m), 1.5*ones(m), n1, 1.5*ones(m), ones(m)], dtype=float).transpose()
    #n = np.array([ones(m), n1, n1, n1, ones(m)], dtype=np.complex128).transpose()

    try:
        selected_freqs_idx = array([np.argwhere(np.isclose(freq, freqs_full))[0][0] for freq in freqs])
    except IndexError:
        print("Check refractive index...")
        m = len(freqs)
        return np.array([ones(m), 1.50 * ones(m), 0.5*(n_min+n_max)*ones(m), 1.50 * ones(m), ones(m)],
                        dtype=float).transpose()

    return n[selected_freqs_idx, :]

def get_n_no_dispersion(freqs, n1=2.70):
    m = len(freqs)

    return np.array([ones(m), 1.5*ones(m), n1*ones(m), 1.5*ones(m), ones(m)], dtype=np.complex128).transpose()