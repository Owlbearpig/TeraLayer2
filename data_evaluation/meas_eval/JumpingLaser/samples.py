import numpy as np


class Sample:
    thickness = None
    layers = None
    tot_thickness = None
    ref_idx = None
    s_id = None

    def __init__(self, thicknesses: list, ref_idx=None, s_id=None):
        # list thicknesses given in mm, stored in um. (Dicke 1 [mm] in case of single layer samples)
        self.thicknesses = np.array(thicknesses) * 1e3  # treat "OP" samples as single layers (ignoring iron core)
        self.tot_thickness = np.sum(self.thicknesses)
        self.layers = len(thicknesses)
        if ref_idx is not None:
            self.ref_idx = np.array(ref_idx, dtype=float)
        self.s_id = s_id


empty = Sample([0.0], s_id=0)
blue_cube = Sample([30.000], [1.54], s_id=1)
fpSample2 = Sample([4.000], [1.65], s_id=2)
fpSample3 = Sample([1.150], [1.65], s_id=3)
fpSample5Plastic = Sample([5.200], s_id=4)
fpSample5ceramic = Sample([1.600], [3.10], s_id=5)
fpSample6 = Sample([0.600], [1.45], s_id=6)
opBluePos1 = Sample([0.210], s_id=7)
opBluePos2 = Sample([0.295], s_id=8)
opBlackPos1 = Sample([0.145], s_id=9)
opBlackPos2 = Sample([0.210], s_id=10)
opRedPos1 = Sample([0.235], s_id=11)
opRedPos2 = Sample([0.335], s_id=12)
opDarkRedPos1 = Sample([0.285], s_id=13)
opDarkRedPos2 = Sample([0.385], s_id=14)
bwCeramicWhiteUp = Sample([0.500, 0.140], s_id=15)
bwCeramicBlackUp = Sample([0.140, 0.500], s_id=16)
ampelMannRight = Sample([0.042, 0.641, 0.074], s_id=17)
ampelMannLeft = Sample([0.074, 0.641, 0.042], s_id=18)
opToolRedPos1 = Sample([0.235], s_id=19)
opToolRedPos2 = Sample([0.335], s_id=20)
opToolBluePos1 = Sample([0.210], s_id=21)
opToolBluePos2 = Sample([0.295], s_id=22)
