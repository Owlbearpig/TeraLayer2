from pathlib import Path
import matplotlib.pyplot as plt

path = Path("Y:\MEGA cloud\AG\TeraLayer\Implementation\FPGAs\HHI LIA\Wave_Generator\Sin_Table.txt")

with open(path, 'r') as file:
    sine_ = []
    for line in file:
        sine_.append(int(line, 2))

plt.plot(sine_)
plt.show()
