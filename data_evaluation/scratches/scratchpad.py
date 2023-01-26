from pathlib import Path
from consts import *
import matplotlib.pyplot as plt
import numpy as np
from numfi import numfi as numfi_
from functools import partial
from numpy import pi as pi64
from numpy import array
from mpl_settings import *


def extract_lists(s):
    ret = []

    in_str = False
    lst_str = ""
    for char in s:
        if in_str:
            lst_str += char
        if char == "]":
            in_str = False
        if char == "[":
            in_str = True

    lists = lst_str.split("]")[:-1]
    for lst in lists:
        formatted_lst = []
        for num in lst.split(" "):
            if num:
                formatted_lst.append(float(num))

        ret.append(formatted_lst)

    return ret


result_dir = ROOT_DIR / Path("optimization") / "results"

results = []
for file in result_dir.glob("*/*.*"):
    with open(file) as f:
        # dir name: /home/alex/PycharmProjects/TeraLayer2/data_evaluation/optimization/results/FP_pd4_p17_mod
        info = str(file.parent).split("4_p")[1].split("_")
        p, dtype = int(info[0]), info[1]

        # file names: FP_results_nm_grid_cw_v1, FP_results_nm_grid_0.75noise_v2
        noise_factor = None
        if "noise" in str(file.stem):
            noise_factor = float(str(file.stem).replace("noise", "").split("_")[-2])

        lines = f.readlines()
        for i, line in enumerate(lines):
            if (i == 104) and (dtype == "cw"):
                val = extract_lists(line)
                res = [p, dtype, noise_factor, *val]
            elif (i == 103) and (dtype == "mod"):
                val = line.replace("\n", "").split(" __ ")
                res = [p, dtype, noise_factor, float(val[0]), int(val[1])]
            else:
                continue
        results.append(res)

results = sorted(results, key=lambda x: x[0])
ps, fails, avg_dev = [], [], []
for res in results:
    #if (res[1] == "mod") and (np.isclose(res[2], 0)):
    if res[1] == "mod" and np.isclose(res[2], 0.75):
        print(res)
        ps.append(res[0])
        fails.append(res[4])
        avg_dev.append(res[3])

avg_dev_label = "Avg. deviation: \n $\\frac{1}{100}\sum_{i=0}^{100} \sum |x_{res,i} - x_{sol,i}| $"
fails_label = "Fail count:\n (Max. dev. above 15 $\mu m)$"

plt.plot(ps, fails, label=fails_label)
plt.plot(ps, avg_dev, label=avg_dev_label)
plt.title("Evaluation of 100 model data sets for different precisions")
plt.xticks(ps)
plt.xlabel("Fractional part length (# of bits)")
plt.legend()
plt.show()

