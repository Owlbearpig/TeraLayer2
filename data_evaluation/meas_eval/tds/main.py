import matplotlib.pyplot as plt
from functions import do_fft
from consts import *
import pandas as pd

"""
goal was to evaluate the refractive index as a function of frequency
but without reference measurement that's probably not possible.
"""

data_dir = Path(ROOT_DIR / "data" / "T-Sweeper_and_TeraFlash" / "Lackierte Keramik" / "Puls (TeraFlash)")

bk_gnd_file = data_dir / "2020_01_30_Background_1000pulses.csv"
data_file = data_dir / "2020_02_20_Ampelmaennchenkopf_Blickrichtung_rechts_1AVG.csv"

bk_gnd_data = pd.read_csv(bk_gnd_file).values
meas_data = pd.read_csv(data_file).values

def main():
    t = bk_gnd_data[:, 0]
    print(meas_data.shape)
    print(bk_gnd_data.shape)
    #for i in range(99):
    plt.figure()
    plt.plot(t, meas_data[98, :])
    plt.plot(t, bk_gnd_data[:, 1])

    plt.figure()
    f, y = do_fft(t, meas_data[98, :])
    plt.plot(f, np.abs(y))


if __name__ == '__main__':
    main()
    plt.show()
