import numpy as np
from enum import Enum


class Sample:
    thickness = None
    layers = None
    tot_thickness = None
    ref_idx = None
    s_id = None

    def __init__(self, thicknesses: list, ref_idx=None, s_id=None):
        # list thicknesses given in mm then converted to um. (Dicke 1 [mm] in case of single layer samples)
        self.thicknesses = np.array(thicknesses) * 1e3  # treat "OP" samples as single layers (ignoring iron core)
        self.tot_thickness = np.sum(self.thicknesses)
        self.layers = len(thicknesses)
        if ref_idx is not None:
            self.ref_idx = np.array(ref_idx, dtype=float)
        else:
            self.ref_idx = np.array(1.5 * np.ones_like(self.thicknesses), dtype=float)
        self.s_id = s_id

    def __repr__(self):
        return f"{self.s_id}"


class SamplesEnum(Enum):
    empty = Sample([0.0], s_id=0)
    blueCube = Sample([30.000], [1.54], s_id=1)
    fpSample2 = Sample([4.000], [1.785], s_id=2)
    fpSample3 = Sample([1.150], [1.65], s_id=3)
    fpSample5Plastic = Sample([5.200], [1.37], s_id=4)
    fpSample5ceramic = Sample([1.600], [2.3], s_id=5)
    fpSample6 = Sample([0.600], [1.35], s_id=6)
    opBluePos1 = Sample([0.210], [1.93], s_id=7)
    opBluePos2 = Sample([0.295], [2.25], s_id=8)
    opBlackPos1 = Sample([0.145], [1.93], s_id=9)
    opBlackPos2 = Sample([0.210], [1.93], s_id=10)
    opRedPos1 = Sample([0.235], [1.93], s_id=11)
    opRedPos2 = Sample([0.335], [1.93], s_id=12)
    opDarkRedPos1 = Sample([0.285], [1.93], s_id=13)
    opDarkRedPos2 = Sample([0.385], [1.93], s_id=14)
    opToolRedPos1 = Sample([0.235], [1.93], s_id=20)
    opToolRedPos2 = Sample([0.335], [1.93], s_id=21)
    opToolBluePos1 = Sample([0.210], [1.93], s_id=22)
    opToolBluePos2 = Sample([0.295], [2.25], s_id=23)
    bwCeramicWhiteUp = Sample([0.500, 0.140], s_id=15)
    bwCeramicBlackUp = Sample([0.140, 0.500], s_id=16)
    ampelMannRight = Sample([0.042, 0.641, 0.074], s_id=17)
    ampelMannLeft = Sample([0.074, 0.641, 0.042], s_id=18)
    ampelMannOld = Sample([0.042, 0.641, 0.074], s_id=19)


if __name__ == '__main__':
    print(SamplesEnum.blueCube.value.s_id)
