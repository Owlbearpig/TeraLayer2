from meas_eval.Measurements.image import Image
from meas_eval.consts import data_dir
import matplotlib.pyplot as plt


def main():
    options = {"en_window": False}

    image = Image(data_dir, options=options)
    image.plot_image()

    samples = [("Sheet1_(1)", 250, (10, 7)),
               ("Sheet2_(4)", 244, (10, -15)),
               ("Glue1_(Sugar_2)", 320, (30, 5)),
               ("Glue2_(Plate1_3)", 171, (49, 7)),
               ("Glue3_(Plate2_6)", 90, (50, -10)),
               # ("3x_Layer_(5)", 521, (32, -12)),
               ]

    for sample in samples:
        if "TripleLayer_5" not in sample[0]:
            pass
        thickness = sample[1]
        label = sample[0] + f" (d={thickness} um)"
        point = sample[2]

        image.plot_point(*point, label=label)

        image.evaluate_point(point, thickness, label, en_plot=True)

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
