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

    def get_ref_idx(self, freq_axis):
        one = np.ones_like(freq_axis)
        n0 = self.ref_idx
        sample_ref_idx = n0[:, np.newaxis] * one

        for i in range(self.layers):
            fa_idx, fe_idx = np.argmin(np.abs(freq_axis - 0.001)), np.argmin(np.abs(freq_axis - 2.000))
            n_r = np.linspace(n0[i].real, n0[i].real * 1.10, fe_idx - fa_idx)
            n_i = np.linspace(n0[i].imag, n0[i].imag * 1.20, fe_idx - fa_idx)
            sample_ref_idx[i, fa_idx:fe_idx] = n_r + 1j*n_i

        if self.has_iron_core:
            n_fe = (500 + 500j) * one
            n = np.array([one, *sample_ref_idx, n_fe, one], dtype=complex).T
        else:
            n = np.array([one, *sample_ref_idx, one], dtype=complex).T

        return n


class SamplesEnum(Enum):
    empty = Sample([0.0])
    blueCube = Sample([30.000], [1.54-0.005j])
    fpSample2 = Sample([4.000], [1.785-0.012j])
    fpSample3 = Sample([1.150], [1.60])
    fpSample5Plastic = Sample([5.200], [1.37])
    fpSample5ceramic = Sample([1.600], [2.31])
    fpSample6 = Sample([0.600], [1.35])
    opBluePos1 = Sample([0.210], [1.93], True)
    opBluePos2 = Sample([0.295], [2.25], True)
    opBlackPos1 = Sample([0.145], [1.93], True)
    opBlackPos2 = Sample([0.210], [1.93], True)
    opRedPos1 = Sample([0.235], [1.93], True)
    opRedPos2 = Sample([0.335], [1.93], True)
    opDarkRedPos1 = Sample([0.285], [1.93], True)
    opDarkRedPos2 = Sample([0.385], [1.93], True)
    opToolRedPos1 = Sample([0.235], [1.93], True)
    opToolRedPos2 = Sample([0.335], [1.93], True)
    opToolBluePos1 = Sample([0.210], [1.93], True)
    opToolBluePos2 = Sample([0.295], [2.25], True)
    bwCeramicWhiteUp = Sample([0.500, 0.140], [2.78-0.015j, 1.52])
    bwCeramicBlackUp = Sample([0.140, 0.500], [2.78-0.015j, 1.52])
    ampelMannRight = Sample([0.046, 0.660, 0.073], [1.52, 2.78-0.015j, 1.52])
    ampelMannLeft = Sample([0.073, 0.660, 0.046], [1.52, 2.78-0.015j, 1.52])
    ampelMannOld = Sample([0.046, 0.660, 0.073], [1.52, 2.78-0.015j, 1.52])


if __name__ == '__main__':
    print(SamplesEnum.blueCube.value.s_id)
