import numpy as np
from numpy import cos, sin, arcsin, exp, dot, conj, pi
from consts import um_to_m
from model.multir_numba import multir_numba


def multir_complex(lam, p):
    thea = 8.0 * pi / 180.0
    n = np.array([1.0+0j, 1.50+0j, 2.8+0j, 1.50+0j, 1.0+0j])
    #n = np.array([1.0, 1.50, 2.8, 1.50, 1.0])
    es = p.copy()
    the = np.zeros(len(lam), dtype=np.complex128)  # np.array([thea, 0, 0, 0, 0, 0])
    ra, rb = np.zeros(len(lam), dtype=np.complex128), np.zeros(len(lam), dtype=np.complex128)
    ta, tb = np.zeros(len(lam), dtype=np.complex128), np.zeros(len(lam), dtype=np.complex128)
    the[0] = thea

    R = np.zeros(len(lam), dtype=np.complex128)
    nc = 3
    for h in range(len(lam)):
        for k in range(nc + 1):
            the[k + 1] = arcsin(n[k] * sin(the[k]) / n[k + 1])
            ra[k] = ((n[k] * cos(the[k + 1])) - ((n[k + 1]) * cos(the[k]))) / (
                    (n[k + 1] * cos(the[k])) + (n[k] * cos(the[k + 1])))
            rb[k] = ((n[k + 1] * cos(the[k])) - (n[k] * cos(the[k + 1]))) / (
                    (n[k] * cos(the[k + 1])) + (n[k + 1] * cos(the[k])))
            ta[k] = (2 * n[k] * cos(the[k + 1])) / ((n[k + 1] * cos(the[k])) + (n[k] * cos(the[k + 1])))
            tb[k] = (2 * n[k + 1] * cos(the[k])) / ((n[k] * cos(the[k + 1])) + (n[k + 1] * cos(the[k])))

        M = (1 / tb[0]) * np.array([[(ta[0] * tb[0]) - (ra[0] * rb[0]), rb[0]],
                                    [-ra[0], 1]], dtype=np.complex128)

        fi = np.zeros(nc, dtype=np.complex128)
        for s in range(nc):
            fi[s] = (2 * pi * n[s + 1] * es[s]) / lam[h]
            Q = (1 / tb[s + 1]) * np.array([[(ta[s + 1] * tb[s + 1]) - (ra[s + 1] * rb[s + 1]), rb[s + 1]],
                                            [-ra[s + 1], 1]], dtype=np.complex128)
            P = np.array([[exp(-fi[s] * 1j), 0], [0, exp(fi[s] * 1j)]])
            M = dot(dot(Q, P), M)

        tt = 1 / M[1, 1]
        rt = M[0, 1] * tt
        R[h] = rt * conj(rt)

    return R


if __name__ == '__main__':
    from data_evaluation.visualizing.plotting import plot_result
    from data_evaluation.consts import custom_mask_420
    from data_evaluation.functions import format_data

    sample_file_idx = 65
    lam, R0 = format_data(sample_file_idx=sample_file_idx)

    p = np.array([45.8205965, 627.06655101, 45.81992758]) * um_to_m

    R_C = multir_complex(lam, p)
    R_R = multir_numba(lam, p)

    print(f"R_C: {R_C}")
    print(f"Sum R complex version: {np.sum(R_C)}")
    print(f"R_R: {R_R}")
    print(f"Sum R real version: {np.sum(R_R)}")

    plot_result(p, multir_complex, mask=custom_mask_420, sample_file_idx=sample_file_idx)
