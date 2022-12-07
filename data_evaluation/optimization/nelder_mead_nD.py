import matplotlib.pyplot as plt
from consts import custom_mask_420, um_to_m, THz, GHz, um
import numpy as np
from numpy import array, sum
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
    def __init__(self, x=None, fx=None, name=""):
        if x is None:
            self.x = np.zeros(n)
        else:
            self.x = x

        self.fx = fx
        self.name = name

    def __repr__(self):
        if self.name:
            return f"{self.name}, x: {self.x}, fx: {self.fx}"
        else:
            return f"x: {self.x}, fx: {self.fx}"


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
            s += sum(simplex.p[i].x[j]) / n
        p_ce.x[j] = s


def update_point(simplex, p_ce, lambda_, p):
    p.x = (1 + lambda_) * p_ce.x - lambda_ * simplex.p[n].x


def simplex_sort(simplex):
    simplex.p = sorted(simplex.p, key=lambda p: p.fx)
    for i, p in enumerate(simplex.p):
        p.name = f"p{i}"


def initial_simplex(p_start, cost_func=None, sample_idx=None, fevals=0, size=0.80):
    simplex = Simplex(*[Point(name=f"p{i}") for i in range(n + 1)])
    for i in range(n + 1):
        for j in range(n):
            if i - 1 == j:
                if not np.isclose(p_start.x[j], 0):
                    simplex.p[i].x[j] = size * p_start.x[j]
                else:
                    simplex.p[i].x[j] = 0.00025
            else:
                simplex.p[i].x[j] = p_start.x[j]
        if cost_func is not None:
            cost_func(simplex.p[i], sample_idx)
            fevals += 1

    if cost_func is not None:
        simplex_sort(simplex)

    return simplex


def grid(p_center=array([150, 600, 150]), spacing=50):
    size = 3
    grid_points = []

    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                point = [p_center[0] + i * spacing, p_center[1] + j * spacing, p_center[2] + k * spacing]
                if all(point):
                    grid_points.append(point)

    return grid_points


def nm_gridsearch(cost_func, p0, options):
    def terminate(iter_cnt, max_iterations, fx):
        # if we reach a good fx val continue iterations for a little longer
        if fx < 0.5:
            return iter_cnt < max_iterations
        else:
            return iter_cnt < max_iterations

    total_iters = 0
    grid_spacing = options["grid_spacing"]
    verbose = False
    iterations = options["iterations"]
    p0_grid = grid(p0, grid_spacing)
    res = {"fun": np.inf, "nfev": 0, "local_fun": []}
    for start_val in p0_grid:
        p_start = Point(array(start_val), name="Start point")
        if verbose:
            print(p_start)

        times_shrinkd = 0
        # RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5 # original values
        RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5
        # chi*rho expand,

        p_r = Point(name="p_r")
        p_e = Point(name="p_e")
        p_c = Point(name="p_c")
        p_ce = Point(name="p_ce")

        cost_func(p_start)
        res["nfev"] += 1
        simplex = initial_simplex(p_start, cost_func, fevals=res["nfev"], size=options["simplex_scale"])
        get_centroid(simplex, p_ce)
        if verbose:
            print("initial simplex and centroid:")
            print(simplex)
            print(p_ce, "\n")

        fx_vals, h = [], 0
        # for h in range(0, iterations):
        while terminate(h, iterations, simplex.p[0].fx):
            h += 1
            shrink = False
            if verbose:
                print(f"start of iteration {h}")
            update_point(simplex, p_ce, RHO, p_r)
            cost_func(p_r)
            res["nfev"] += 1

            if p_r.fx < simplex.p[0].fx:
                if verbose:
                    print("difference p_r.fx < simplex.p[0].fx", f'{abs(p_r.fx - simplex.p[0].fx):.20f}')
                update_point(simplex, p_ce, RHO * CHI, p_e)
                cost_func(p_e)
                res["nfev"] += 1
                if p_e.fx < p_r.fx:
                    if verbose:
                        print("difference p_e.fx < p_r.fx", f'{abs(p_e.fx - p_r.fx):.20f}')
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
                              f'{abs(p_r.fx - simplex.p[n - 1].fx):.20f}')
                        print("reflect 2")
                    copy_point(p_r, simplex.p[n])
                else:
                    if p_r.fx < simplex.p[n].fx:
                        if verbose:
                            print("difference p_r.fx < simplex.p[3].fx",
                                  f'{abs(p_r.fx - simplex.p[n].fx):.20f}')
                        update_point(simplex, p_ce, RHO * GAMMA, p_c)
                        cost_func(p_c)
                        res["nfev"] += 1
                        if p_c.fx <= p_r.fx:
                            if verbose:
                                print("difference p_c.fx <= p_r.fx", f'{abs(p_c.fx - p_r.fx):.20f}')
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
                                      f'{abs(p_c.fx - simplex.p[n].fx):.20f}')
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
            print(f"times shrinked: {times_shrinkd}")
            print("final point values: ")
            print(p_r)
            print(p_e)
            print(p_c)
            print(p_ce)
            print(simplex, "\n")

        # iterations, start value
        print(h, start_val)
        total_iters += h
        # solution in p0 of simplex
        print("solution (simplex.p0):", simplex.p[0], "\n")
        res["local_fun"].append(simplex.p[0].fx)
        if simplex.p[0].fx < res["fun"]:
            res["x"], res["fun"], res["lstart"] = simplex.p[0].x, simplex.p[0].fx, start_val

    if verbose:
        print("Best minimum: ", np.round(res["x"], 2), res["fun"])
    res["total_iters"] = total_iters

    return res


if __name__ == '__main__':
    p0 = array([150, 600, 150])
    grid_spacing = 50
    grid_points = grid(p0, grid_spacing)

    print(grid_points)
    print(len(grid_points))
