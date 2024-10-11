import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from consts import ROOT_DIR, selected_freqs
from mpl_settings import *


def main():
    teralyzer_results_dir = ROOT_DIR / "meas_eval" / "Gluesheets" / "teralyzer_results"

    csv_files = [file for file in os.listdir(teralyzer_results_dir) if "csv" in str(file)]

    names = []
    for csv_file in csv_files:
        pos_x, pos_y = None, None
        if "sheet" in str(csv_file):
            name, d, _ = str(csv_file).split("_")
        else:
            name, pos_x, pos_y, d, _ = str(csv_file).split("_")

        # print(name, pos_x, pos_y, d)
        pd_df = pd.read_csv(teralyzer_results_dir / csv_file)

        freq_key = [key for key in pd_df.keys() if "freq" in key][0]
        ref_ind_key = [key for key in pd_df.keys() if "ref_ind" in key][0]
        alpha_key = [key for key in pd_df.keys() if "alpha" in key][0]
        freqs = pd_df[freq_key] / 1e12
        n = pd_df[ref_ind_key]
        alpha = pd_df[alpha_key]

        label = f"{name} {pos_x} {pos_y} d={d} um"

        if "sheet" in name:
            plt.figure("Refractive index sheets")
            plt.title("Sheet samples")
        else:
            plt.figure("Refractive index glue")
            plt.title("Glue samples")
        if "3" in name:
            plt.plot(freqs, n, label=label, ls="dotted")
        else:
            plt.plot(freqs, n, label=label)

        plt.xlabel("Frequency (THz)")
        plt.ylabel("Refractive index")

        if "sheet" in name:
            plt.figure("Absorption coefficient sheets")
            plt.title("Sheet samples")
        else:
            plt.figure("Absorption coefficient glue")
            plt.title("Glue samples")

        if "3" in name:
            plt.plot(freqs, alpha, label=label, ls="dotted")
        else:
            plt.plot(freqs, alpha, label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Absorption coefficient (1/cm)")

        if "3" in name:
            continue
        print(selected_freqs)
        for s_freq in selected_freqs:
            print(s_freq)
            if s_freq < freqs[0]:
                n_fit = np.polyfit(freqs, n, 1)
                line = np.poly1d(n_fit)
                n_extr = line(s_freq)
                alpha_fit = np.polyfit(freqs, alpha, 1)
                line = np.poly1d(alpha_fit)
                alpha_extr = line(s_freq)
                print("(extrapolated)", name, f"{pos_x}_{pos_y}", n_extr, alpha_extr, "\n")
            else:
                freq_idx = np.argmin(np.abs(freqs - s_freq))
                print(name, f"{pos_x}_{pos_y}", n[freq_idx], alpha[freq_idx], "\n")

        names.append(f"{name} {pos_x}_{pos_y}")

    print(names)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            plt.legend()

        axes = fig.get_axes()
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend()
    plt.show()


if __name__ == '__main__':
    main()

