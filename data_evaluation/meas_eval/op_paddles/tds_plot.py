import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from mpl_settings import *

data_dir = Path(r"E:\measurementdata\TeraLayer2\OP-Besteck\2023-05-16_OP-Proben_Paddles")

csv_files = [file for file in os.listdir(data_dir) if "csv" in str(file)]

for csv_file in csv_files:
    data_file = data_dir / csv_file
    pd_df = pd.read_csv(data_file)
    data = np.array(pd_df)

    if "ref" in str(csv_file):
        plt.plot(data[:, 0], data[:, 1], label="Reference")
    if "Nr10_2_Trocken" in str(csv_file):
        plt.plot(data[:, 0], data[:, 1], label="Nr10_2_Trocken")
    if "Nr2_Trocken" in str(csv_file):
        plt.plot(data[:, 0], data[:, 1], label="Nr2_Trocken")

plt.xlabel("time (ps)")
plt.ylabel("amplitude (arb. u.)")
plt.legend()
plt.show()

