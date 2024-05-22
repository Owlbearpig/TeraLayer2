import matplotlib as mpl
from meas_eval.consts import os_name, Path, result_dir
from typing import Dict
import matplotlib.font_manager


# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])

def fmt(x, val):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    # return r'${} \times 10^{{{}}}$'.format(a, b)
    return rf'{a}E+{b:02}'


def mpl_style_params(new_rcparams=None):
    rcParams = mpl.rcParams
    # rcParams['lines.linestyle'] = '--'
    # rcParams['legend.fontsize'] = 'large' #'x-large'
    rcParams['legend.shadow'] = False
    rcParams['lines.marker'] = 'o'
    rcParams['lines.markersize'] = 2
    # rcParams['lines.linewidth'] = 3.5  # 2
    rcParams['ytick.major.width'] = 2.5
    rcParams['xtick.major.width'] = 2.5
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['axes.grid'] = True
    rcParams['figure.autolayout'] = False
    rcParams['savefig.format'] = 'png'
    rcParams["scatter.marker"] = "o"  # "x"
    rcParams.update({'font.size': 16})

    # Say, "the default sans-serif font is COMIC SANS"
    # rcParams['font.sans-serif'] = 'Liberation Sans'
    # Then, "ALWAYS use sans-serif fonts"
    # rcParams['font.family'] = "sans-serif"

    rcParams["savefig.directory"] = result_dir

    if new_rcparams:
        rcParams.update(new_rcparams)

    Path(rcParams["savefig.directory"]).mkdir(parents=True, exist_ok=True)

    return rcParams

"""
from matplotlib.pyplot import subplots, xlabel, ylabel, grid, show
fig, ay = subplots()

# Using the specialized math font elsewhere, plus a different font
xlabel(r"The quick brown fox jumps over the lazy dog", fontsize=18)
# No math formatting, for comparison
ylabel(r'Italic and just Arial and not-math-font', fontsize=18)
grid()

show()
"""

if __name__ == '__main__':
    rcParams = mpl_style_params()
    print(rcParams.keys())
