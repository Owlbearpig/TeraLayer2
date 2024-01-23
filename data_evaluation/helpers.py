import warnings
from typing import Callable, Iterable
from scipy.optimize import root_scalar
import numpy as np
from pathlib import Path


def is_iterable(obj):
    """
    print(is_iterable(3.4))
    print(is_iterable([3.4]))
    """
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


def multi_root(f: Callable, bracket: Iterable[float], args: Iterable = (), n: int = 30) -> np.ndarray:
    """ Find all roots of f in `bracket`, given that resolution `n` covers the sign change.
    Fine-grained root finding is performed with `scipy.optimize.root_scalar`.
    Parameters
    ----------
    f: Callable
        Function to be evaluated
    bracket: Sequence of two floats
        Specifies interval within which roots are searched.
    args: Iterable, optional
        Iterable passed to `f` for evaluation
    n: int
        Number of points sampled equidistantly from bracket to evaluate `f`.
        Resolution has to be high enough to cover sign changes of all roots but not finer than that.
        Actual roots are found using `scipy.optimize.root_scalar`.
    Returns
    -------
    roots: np.ndarray
        Array containing all unique roots that were found in `bracket`.
    """
    # Evaluate function in given bracket
    x = np.linspace(*bracket, n)
    y = f(x, *args)

    # Find where adjacent signs are not equal
    sign_changes = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]

    # Find roots around sign changes
    root_finders = (
        root_scalar(
            f=f,
            args=args,
            bracket=(x[s], x[s + 1])
        )
        for s in sign_changes
    )

    roots = np.array([
        r.root if r.converged else np.nan
        for r in root_finders
    ])

    if np.any(np.isnan(roots)):
        warnings.warn("Not all root finders converged for estimated brackets! Maybe increase resolution `n`.")
        roots = roots[~np.isnan(roots)]

    roots_unique = np.unique(roots)
    if len(roots_unique) != len(roots):
        warnings.warn("One root was found multiple times. "
                      "Try to increase or decrease resolution `n` to see if this warning disappears.")

    return roots_unique


def save_fig(fig_num_, mpl=None, save_dir=None, filename=None, **kwargs):
    if mpl is None:
        import matplotlib as mpl

    plt = mpl.pyplot

    rcParams = mpl.rcParams

    if save_dir is None:
        save_dir = Path(rcParams["savefig.directory"])

    fig = plt.figure(fig_num_)

    if filename is None:
        filename_s = str(fig.canvas.get_window_title())
    else:
        filename_s = str(filename)

    unwanted_chars = ["(", ")"]
    for char in unwanted_chars:
        filename_s = filename_s.replace(char, '')
    filename_s.replace(" ", "_")

    fig.set_size_inches((16, 9), forward=False)
    plt.savefig(save_dir / (filename_s + ".png"), bbox_inches='tight', dpi=300, pad_inches=0, **kwargs)


def plt_show(mpl_, en_save=False):
    plt_ = mpl_.pyplot
    for fig_num in plt_.get_fignums():
        fig = plt_.figure(fig_num)
        for ax in fig.get_axes():
            h, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend()

        if en_save:
            save_fig(fig_num, mpl_)
    plt_.show()


if __name__ == '__main__':
    from functools import partial
    from scratches.scratch_sheet import f

    f = partial(f, d2_=45)

    roots = multi_root(f, [0, 200])
    print(roots)
