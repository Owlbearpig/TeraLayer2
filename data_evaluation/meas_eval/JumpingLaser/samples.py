import numpy as np
from enum import Enum


class Sample:
    thickness = None
    layers = None
    tot_thickness = None
    ref_idx = None
    s_id = None

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
        return f"{self.s_id}"


class SamplesEnum(Enum):
    empty = Sample([0.0])
    blueCube = Sample([30.000], [1.54])
    fpSample2 = Sample([4.000], [1.785])
    fpSample3 = Sample([1.150], [1.60])
    fpSample5Plastic = Sample([5.200], [1.37])
    fpSample5ceramic = Sample([1.600], [2.3])
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
    bwCeramicWhiteUp = Sample([0.500, 0.140])
    bwCeramicBlackUp = Sample([0.140, 0.500])
    ampelMannRight = Sample([0.042, 0.641, 0.074], [1.52, 2.78, 1.52])
    ampelMannLeft = Sample([0.074, 0.641, 0.042], [1.52, 2.78, 1.52])
    ampelMannOld = Sample([0.042, 0.641, 0.074], [1.52, 2.78, 1.52])


if __name__ == '__main__':
    print(SamplesEnum.blueCube.value.s_id)
