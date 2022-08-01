import numpy as np
from numpy import zeros
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
from scipy.signal import windows
import os

MHz = 10 ** 6
THz = 10 ** 12

if os.name == 'posix':
    dir_path = Path(r"")
else:
    dir_path = Path("E:\Projects\TeraLayer2\data_evaluation\matlab_enrique\Data")

file_paths = [Path(root) / file for root, dirs, files in os.walk(dir_path) for file in files]

data_array = np.array([pd.read_csv(file).values for file in file_paths if "Kopf" in str(file)])

print(data_array.shape)

avg_data = np.mean(data_array, axis=0)

plt.plot(avg_data[:, 0]*MHz, avg_data[:, 2])
plt.show()
