import numpy as np

from consts import *
from numpy import cos, sin, arcsin, exp, dot, conj

def multir(lam, p):
    es = p.copy()
    the = np.zeros(len(n), dtype=np.complex128)
    ra, rb, ta, tb = np.zeros((4, len(n)-1), dtype=np.complex128)
    the[0] = thea

    R = np.zeros(len(lam), dtype=np.complex128)
    nc = len(n)-2
    for h in range(len(lam)):
        for k in range(nc+1):
            the[k + 1] = arcsin(n[k] * sin(the[k]) / n[k+1])
            if a == 1:
                ra[k] = ((n[k] * cos(the[k + 1])) - ((n[k + 1]) * cos(the[k]))) / (
                            (n[k + 1] * cos(the[k])) + (n[k] * cos(the[k + 1])))
                rb[k] = ((n[k + 1] * cos(the[k])) - (n[k] * cos(the[k + 1]))) / (
                            (n[k] * cos(the[k + 1])) + (n[k + 1] * cos(the[k])))
                ta[k] = (2 * n[k] * cos(the[k + 1])) / ((n[k + 1] * cos(the[k])) + (n[k] * cos(the[k + 1])))
                tb[k] = (2 * n[k + 1] * cos(the[k])) / ((n[k] * cos(the[k + 1])) + (n[k + 1] * cos(the[k])))
            else:
                ra[k] = ((n[k] * cos(the[k])) - (n[k + 1] * cos(the[k + 1]))) / (
                            (n[k] * cos(the[k])) + (n[k + 1] * cos(the[k + 1])))
                rb[k] = ((n[k + 1] * cos(the[k + 1])) - (n[k] * cos(the[k]))) / (
                            (n[k + 1] * cos(the[k + 1])) + (n[k] * cos(the[k])))
                ta[k] = (2 * n[k] * cos(the[k])) / ((n[k] * cos(the[k])) + (n[k + 1] * cos(the[k + 1])))
                tb[k] = (2 * n[k + 1] * cos(the[k + 1])) / ((n[k + 1] * cos(the[k + 1])) + (n[k] * cos(the[k])))

        M = (1 / tb[0]) * np.array([[(ta[0] * tb[0]) - (ra[0] * rb[0]), rb[0]],
                                    [-ra[0], 1]], dtype=np.complex128)

        fi = np.zeros(nc, dtype=np.complex128)
        for s in range(nc):
            fi[s] = (2 * pi * n[s + 1] * es[s]) / lam[h] # could be that es -> es*cos(the) ?
            Q = (1 / tb[s + 1]) * np.array([[(ta[s + 1] * tb[s + 1]) - (ra[s + 1] * rb[s + 1]), rb[s + 1]],
                                            [-ra[s + 1], 1]], dtype=np.complex128)
            P = np.array([[exp(-fi[s] * 1j), 0], [0, exp(fi[s] * 1j)]], dtype=np.complex128)
            M = dot(dot(Q, P), M)

        tt = 1 / M[1, 1]
        rt = M[0, 1] * tt

        R[h] = (rt * conj(rt)).real

    return conj(R)

if __name__ == '__main__':
    from functions import format_data
    import matplotlib.pyplot as plt
    # lam, R = format_data(mask=full_range_mask)

    x_axis = np.arange(200, 1500, 0.1).astype(np.float64)
    # custom_mask_420 = array([421., 521., 651., 801., 851., 951.])
    x_axis = custom_mask_420.astype(np.float64)
    freqs = x_axis * GHz
    lam = c0 / freqs

    #p0 = array([2860.0, 4997.0, 0.0]) * um_to_m
    p0 = array([2860.0, 0.0, 0.0]) * um_to_m
    R0 = multir(lam, p0)
    print(R0)
    plt.plot(freqs, (np.abs(R0)))
    plt.show()