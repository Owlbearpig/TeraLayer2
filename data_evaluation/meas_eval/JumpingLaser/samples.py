import numpy as np
from enum import Enum


class Sample:
    thickness = None
    layers = None
    tot_thickness = None
    ref_idx = None
    name = None

    def __init__(self, thicknesses: list, ref_idx=None, core=False):
        # list thicknesses given in mm then converted to um. (Dicke 1 [mm] in case of single layer samples)
        self.thicknesses = np.array(thicknesses) * 1e3  # treat "OP" samples as single layers (ignoring iron core)
        self.tot_thickness = np.sum(self.thicknesses)
        self.layers = len(thicknesses)
        if ref_idx is not None:
            self.ref_idx = np.array(ref_idx, dtype=complex)
        else:
            self.ref_idx = np.array(1.5 * np.ones_like(self.thicknesses), dtype=complex)
        self.has_iron_core = core

    def __repr__(self):
        return f"{self.name}"

    def set_thicknesses(self, new_thicknesses):
        # new_thicknesses should be in um
        self.thicknesses = np.array(new_thicknesses, dtype=float)
        self.tot_thickness = np.sum(new_thicknesses)

    def set_ref_idx(self, ref_idx):
        self.ref_idx = np.array(ref_idx, dtype=complex)

    def get_ref_idx(self, selected_freqs=None):
        freq_axis = np.arange(-0.200, 5.000, 0.001)
        one = np.ones_like(freq_axis)
        sample_ref_idx = np.zeros((self.layers, len(freq_axis)), dtype=complex)
        n0 = self.ref_idx

        fa_idx, fe_idx = np.argmin(np.abs(freq_axis - -0.11)), np.argmin(np.abs(freq_axis - 2.000))
        for i in range(self.layers):
            n_min, n_max = n0[i][0], n0[i][1]

            n_r = np.linspace(n_min.real, n_max.real, fe_idx - fa_idx)
            n_i = np.linspace(n_min.imag, n_max.imag, fe_idx - fa_idx)
            sample_ref_idx[i, :fa_idx] = np.ones(fa_idx)
            sample_ref_idx[i, fa_idx:fe_idx] = n_r + 1j * n_i
            sample_ref_idx[i, fe_idx:] = np.ones(len(freq_axis) - fe_idx)

        if self.has_iron_core:
            n_fe = (500 + 500j) * one
            n = np.array([one, *sample_ref_idx, n_fe, one], dtype=complex).T
        else:
            n = np.array([one, *sample_ref_idx, one], dtype=complex).T

        if selected_freqs is not None:
            sel_freq_idx = [np.argmin(np.abs(freq_axis - sel_freq)) for sel_freq in selected_freqs]
            n = n[sel_freq_idx, :]

        return n


class SamplesEnum(Enum):
    empty = Sample([0.0])

    # 1 layer
    blueCube = Sample([30.000], [(1.54 - 0.005j, 1.54 - 0.005j)])  # probably ifft -> window -> fft
    fpSample2 = Sample([4.000], [(1.683 - 0.0192j, 1.699 - 0.024j)])  # WORKS
    fpSample3 = Sample([1.150], [(1.676 - 0.0092j, 1.68 - 0.0945j)])  # WORKS
    fpSample5Plastic = Sample([5.200], [(1.2 - 0.0001j, 1.9 - 0.0080j)])  # doesnt work
    fpSample5ceramic = Sample([1.600], [(2.307 - 0.00334j, 2.330 - 0.012j)])  # WORKS
    fpSample6 = Sample([0.600], [(1.34 - 0.028j, 1.370 - 0.15j)])  # kriege ich nicht gut hin ...

    # 1 layer + mirror core
    opBluePos1 = Sample([0.210], [(1.93, 1.93)], True)
    opBluePos2 = Sample([0.295], [(2.25, 2.25)], True)
    opBlackPos1 = Sample([0.145], [(1.93, 1.93)], True)
    opBlackPos2 = Sample([0.210], [(1.93, 1.93)], True)
    opRedPos1 = Sample([0.235], [(1.93, 1.93)], True)
    opRedPos2 = Sample([0.335], [(1.93, 1.93)], True)
    opDarkRedPos1 = Sample([0.285], [(1.93, 1.93)], True)
    opDarkRedPos2 = Sample([0.385], [(1.93, 1.93)], True)
    opToolRedPos1 = Sample([0.235], [(1.93, 1.93)], True)
    opToolRedPos2 = Sample([0.335], [(1.93, 1.93)], True)
    opToolBluePos1 = Sample([0.210], [(1.93, 1.93)], True)
    opToolBluePos2 = Sample([0.295], [(2.25, 2.25)], True)

    # 2 layer
    bwCeramicWhiteUp = Sample([0.500, 0.140], [(2.911 - 0.001j, 2.950 - 0.059j), (2.685 - 0.001j, 2.722 - 0.0636j)])  # WORKS
    bwCeramicBlackUp = Sample([0.140, 0.500], [(2.685 - 0.001j, 2.722 - 0.0636j), (2.911 - 0.001j, 2.950 - 0.059j)])  # WORKS

    # 3 layer
    ampelMannRight = Sample([0.046, 0.660, 0.073], [(1.527, 1.532), (2.80 - 0.000j, 2.82 - 0.015j),
                                                    (1.527, 1.532)])  # WORKS
    ampelMannLeft = Sample([0.073, 0.660, 0.046], [(1.52, 1.521), (2.78 - 0.000j, 2.78 - 0.015j),
                                                   (1.52, 1.521)])
    ampelMannOld = Sample([0.046, 0.660, 0.073], [(1.52, 1.521), (2.78 - 0.000j, 2.78 - 0.015j),
                                                  (1.52, 1.521)])


if __name__ == '__main__':
    print(SamplesEnum.blueCube.name)
