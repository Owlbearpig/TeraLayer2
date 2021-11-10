import numpy as np
from consts import *
from numpy import cos, sin, asin, sum, exp, dot, conj


# TODO Fix loop index
def multir(p, lam):
    # lam: number of wavelengths (=6)
    es = p.copy()
    the = np.zeros(len(lam))
    ra, rb, ta, tb = np.zeros(len(lam)), np.zeros(len(lam)), np.zeros(len(lam)), np.zeros(len(lam))
    the[0] = thea

    R = np.zeros(len(lam))
    nc = len(n) - 2 # changed
    for h in range(1, sum(len(lam)) - 1):
        for k in range(1, nc + 1):
            the[k + 1] = asin(n[k] * sin(the[k]) / n[k+1])
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
                                    [-ra[0], 1]])

        fi = np.zeros(nc)
        for s in range(1, nc):
            fi[s] = (2 * pi * n[s + 1] * es[s]) / lam[h]
            Q = (1 / tb[s + 1]) * np.array([[(ta[s + 1] * tb[s + 1]) - (ra[s + 1] * rb[s + 1]), rb[s + 1]],
                                            [-ra[s + 1], 1]])
            P = np.array([[exp(-fi[s] * 1j), 0], [0, exp(fi[s] * 1j)]])
            M = dot(dot(Q, P), M)

        tt = 1 / M[1, 1]
        rt = M[0, 1] * tt
        R[h] = rt * conj(rt)

    return conj(R)
