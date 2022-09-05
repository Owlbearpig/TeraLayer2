from pathlib import Path
import numpy as np

path = Path(r"E:\Mega\AG\Wihi job\TeraLayer\Implementation\Notepads\freq_sweep_p1.txt")

min_val, best_freqs = np.inf, None
with open(path) as file:
    for line in file.readlines():
        val = float(line.split(" ")[1])

        if val < min_val:
            min_val = val
            best_freqs = line.split(" ")[2:]

print(min_val, best_freqs)

lines = []
with open(path) as file:
    for line in file.readlines():
        val = float(line.split(" ")[1])
        freqs = line.split(" ")[2:]
        lines.append((val, freqs))

lines = sorted(lines, key=lambda x: x[0])
for i, line in enumerate(lines):
    print(i, line)

