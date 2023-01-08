import matplotlib.pyplot as plt
from consts import custom_mask_420, um_to_m, THz, GHz, um
import numpy as np
from numfi import numfi
from numpy import array, sum, pi
import matplotlib as mpl

# mpl.rcParams['lines.linestyle'] = '--'
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['xtick.major.width'] = 2.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
# plt.style.use(['dark_background'])
# plt.xkcd()
mpl.rcParams.update({'font.size': 22})

n = 3


class Simplex:
    def __init__(self, *args):
        self.p = [*args]

    def __repr__(self):
        s = ""
        for i, p in enumerate(self.p):
            s += str(p) + "\n" * (i != n)
        return s


class Point:
    def __init__(self, x, fx=None, name=""):
        self.x = x
        self.fx = fx
        self.name = name

    def __repr__(self):
        if self.name:
            return f"{self.name}, x: {self.x}, fx: {self.fx}"
        else:
            return f"x: {self.x}, fx: {self.fx}"

    def __getitem__(self, key):
        if key == len(self.x):
            return self.fx
        else:
            return self.x[key]

    def __setitem__(self, key, value):
        if key == len(self.x):
            self.fx = value
        else:
            self.x[key] = value

def swap_points(p1, p2):
    tmp = Point(p1.x, p1.fx)
    p1.x, p1.fx = p2.x, p2.fx
    p2.x, p2.fx = tmp.x, tmp.fx


def copy_point(p_src, p_dst):
    p_dst.x, p_dst.fx = p_src.x, p_src.fx


def get_centroid(simplex, p_ce):
    for j in range(n):
        s = 0
        for i in range(n):
            s += (1 / n) * sum(simplex.p[i].x[j])
        p_ce.x[j] = s


def update_point(simplex, p_ce, lambda_, p):
    p.x = (1 + lambda_) * p_ce.x - lambda_ * simplex.p[n].x


def simplex_sort(simplex):
    simplex.p = sorted(simplex.p, key=lambda p: p.fx)
    for i, p in enumerate(simplex.p):
        p.name = f"p{i}"


def initial_simplex(p_start, cost_func=None, fevals=0, spread=40):
    simplex = Simplex(*[Point(x=0*p_start.x, name=f"p{i}") for i in range(n + 1)])
    for i in range(n + 1):
        for j in range(n):
            if i - 1 == j:
                if isinstance(p_start.x, numfi):
                    simplex.p[i].x[j] = p_start.x[j] - (spread / (2 * pi * 2 ** 6))
                else:
                    simplex.p[i].x[j] = p_start.x[j] - spread
            else:
                simplex.p[i].x[j] = p_start.x[j]
        if cost_func is not None:
            cost_func(simplex.p[i])
            fevals += 1

    if cost_func is not None:
        simplex_sort(simplex)

    return simplex


def grid(p_center, spacing=50, size=3):
    grid_points = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                if isinstance(p_center, numfi):
                    point = p_center + (array([i, j, k]) * (spacing * (1 / (2 * pi * 2 ** 6))))
                else:
                    point = p_center + array([i, j, k]) * spacing

                point[point < 0] = 0
                if all(point) and all([x >= 0 for x in point]):
                    grid_points.append(point)

    return grid_points


def nm_algo(start_val, cost_func, res, options):
    iterations = options["iterations"]

    verbose = options["verbose"]

    p_start = Point(start_val, name="Start point")
    if verbose:
        print(p_start)

    times_shrinkd = 0
    # RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5 # original values
    RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5
    # chi*rho expand,

    p_r = Point(x=0*p_start.x, name="p_r")
    p_e = Point(x=0*p_start.x, name="p_e")
    p_c = Point(x=0*p_start.x, name="p_c")
    p_ce = Point(x=0*p_start.x, name="p_ce")

    cost_func(p_start)
    res["nfev"] += 1
    simplex = initial_simplex(p_start, cost_func, fevals=res["nfev"], spread=options["simplex_spread"])

    get_centroid(simplex, p_ce)
    if verbose:
        print("initial simplex and centroid:")
        print(simplex)
        print(p_ce, "\n")

    fx_vals, h = [], 0
    for h in range(0, iterations):
        h += 1
        shrink = False
        if verbose:
            print(f"start of iteration {h}")
        update_point(simplex, p_ce, RHO, p_r)
        cost_func(p_r)
        res["nfev"] += 1

        if p_r.fx < simplex.p[0].fx:
            if verbose:
                print("difference p_r.fx < simplex.p[0].fx", f'{abs(p_r.fx - simplex.p[0].fx)}')
            update_point(simplex, p_ce, RHO * CHI, p_e)
            cost_func(p_e)
            res["nfev"] += 1
            if p_e.fx < p_r.fx:
                if verbose:
                    print("difference p_e.fx < p_r.fx", f'{abs(p_e.fx - p_r.fx)}')
                    print("expand")
                copy_point(p_e, simplex.p[n])
            else:
                if verbose:
                    print("reflect 1")
                copy_point(p_r, simplex.p[n])
        else:
            if p_r.fx < simplex.p[n - 1].fx:
                if verbose:
                    print("difference p_r.fx < simplex.p[2].fx",
                          f'{abs(p_r.fx - simplex.p[n - 1].fx)}')
                    print("reflect 2")
                copy_point(p_r, simplex.p[n])
            else:
                if p_r.fx < simplex.p[n].fx:
                    if verbose:
                        print("difference p_r.fx < simplex.p[3].fx",
                              f'{abs(p_r.fx - simplex.p[n].fx)}')
                    update_point(simplex, p_ce, RHO * GAMMA, p_c)
                    cost_func(p_c)
                    res["nfev"] += 1
                    if p_c.fx <= p_r.fx:
                        if verbose:
                            print("difference p_c.fx <= p_r.fx", f'{abs(p_c.fx - p_r.fx)}')
                            print("contract out")
                        copy_point(p_c, simplex.p[n])
                    else:
                        if verbose:
                            print("shrink check p_c.fx <= p_r.fx\n")
                        shrink = True
                else:
                    update_point(simplex, p_ce, -GAMMA, p_c)
                    cost_func(p_c)
                    res["nfev"] += 1
                    if p_c.fx <= simplex.p[n].fx:
                        if verbose:
                            print("difference p_c.fx <= simplex.p[3].fx",
                                  f'{abs(p_c.fx - simplex.p[n].fx)}')
                            print("contract in")
                        copy_point(p_c, simplex.p[n])
                    else:
                        if verbose:
                            print("shrink check p_c.fx <= simplex.p3.fx\n")
                        shrink = True
        if shrink:
            times_shrinkd += 1
            for i in range(1, n + 1):
                for j in range(n):
                    simplex.p[i].x[j] = simplex.p[0].x[j] + SIGMA * (
                            simplex.p[i].x[j] - simplex.p[0].x[j])
                cost_func(simplex.p[i])
                res["nfev"] += 1
            simplex_sort(simplex)
        else:
            # insertion sort
            for k in reversed(range(n)):
                if simplex.p[k + 1].fx < simplex.p[k].fx:
                    swap_points(simplex.p[k + 1], simplex.p[k])

        get_centroid(simplex, p_ce)
        if verbose:
            print(p_r)
            print(p_e)
            print(p_c)
            print(p_ce)
            print(simplex)
            print(f"iteration {h} done\n")

        fx_vals.append(simplex.p[0].fx)

    if verbose:
        print(f"Times shrinked: {times_shrinkd}")
        print("Final point values: ")
        print(p_r)
        print(p_e)
        print(p_c)
        print(p_ce)
        print(simplex, "\n")

    # iterations, start value
    print(f"Iterations completed: {h}")
    print(f"Upscaled starting point: {array(start_val) * (2*pi*2**6)}, (original: {start_val})")
    res["total_iters"] += h
    # solution in p0 of simplex
    print("solution (simplex.p0):", simplex.p[0], "\n")
    res["local_fun"].append(simplex.p[0].fx)

    if res["fun"] is None: # if first start val gives best result
        res["x"], res["fun"] = simplex.p[0].x, simplex.p[0].fx
        res["best_start_points"].append(start_val)
    elif simplex.p[0].fx < res["fun"]:
        res["x"], res["fun"] = simplex.p[0].x, simplex.p[0].fx
        if not options["enhance_step"]:
            res["best_start_points"].append(start_val)


def nm_gridsearch(cost_func, p0, options):

    grid_spacing = options["grid_spacing"]
    size = options["size"]

    if not "simplex_spread" in options.keys():
        options["simplex_spread"] = 40

    if "numfi" in options.keys():
        p0 = options["numfi"](p0 * (1 / (2*pi*2**6)))

    p0_grid = grid(p0, grid_spacing, size)

    res = {"fun": None, "nfev": 0, "local_fun": [], "total_iters": 0,
           "best_start_points": [], "p0_dist": []}

    for start_val in p0_grid:
        # perform a single run of the nm algo
        nm_algo(start_val, cost_func, res, options)

    # enhance best results
    options["iterations"] = 3 * options["iterations"]
    options["enhance_step"] = True

    nm_algo(res["best_start_points"][-1], cost_func, res, options)

    if options["verbose"]:
        print("Best minimum: ", res["x"], res["fun"])

    if isinstance(p0, numfi):
        upscale = (2*pi*2**6)
        res["x"] = array(res["x"]) * upscale
        res["best_start_points"] = [array(x) * upscale for x in res["best_start_points"]]

    return res


if __name__ == '__main__':
    p0 = array([150, 600, 150])
    grid_spacing, size = 50, 3
    grid_points = grid(p0, grid_spacing, size)

    print(grid_points)
    print(len(grid_points))

    p_zero = Point(p0)
    print(p_zero[3])