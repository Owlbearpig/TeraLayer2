from model.explicitEvalSimple import explicit_reflectance
from consts import custom_mask_420, um_to_m
from functions import format_data
import numpy as np
from numpy import array, sum


class Simplex:
    def __init__(self, p0=None, p1=None, p2=None, p3=None):
        self.p = [p0, p1, p2, p3]

    def __repr__(self):
        s = ""
        for i, p in enumerate(self.p):
            s += str(p) + "\n"*(i != 3)
        return s


class Point:
    def __init__(self, x=None, fx=None, name=""):
        if x is None:
            self.x = np.zeros(3)
        else:
            self.x = x

        self.fx = fx
        self.name = name

    def __repr__(self):
        return f"{self.name}, x: {self.x}, fx: {self.fx}"


def swap_points(p1, p2):
    tmp = Point(p1.x, p1.fx)
    p1.x, p1.fx = p2.x, p2.fx
    p2.x, p2.fx = tmp.x, tmp.fx


def copy_point(p_src, p_dst):
    p_dst.x, p_dst.fx = p_src.x, p_src.fx


def get_centroid(simplex, p_ce):
    for j in range(3):
        s = 0
        for i in range(3):
            s += sum(simplex.p[i].x[j]) / 3
        p_ce.x[j] = s


def update_point(simplex, p_ce, lambda_, p):
    p.x = (1 + lambda_) * p_ce.x - lambda_ * simplex.p[3].x


def cost(p, sample_idx=10):
    mask = custom_mask_420

    _, R0 = format_data(mask=mask, sample_file_idx=sample_idx, verbose=False)

    p.fx = sum((explicit_reflectance(p.x * um_to_m) - R0) ** 2)


def simplex_sort(simplex):
    simplex.p = sorted(simplex.p, key=lambda p: p.fx)
    for i, p in enumerate(simplex.p):
        p.name = f"p{i}"


def initial_simplex(p_start, only_coords=False):
    n = 3

    simplex = Simplex(Point(name="p0"), Point(name="p1"), Point(name="p2"), Point(name="p3"))
    for i in range(n + 1):
        for j in range(3):
            if i - 1 == j:
                if not np.isclose(p_start.x[j], 0):
                    simplex.p[i].x[j] = 1.05 * p_start.x[j]
                else:
                    simplex.p[i].x[j] = 0.00025
            else:
                simplex.p[i].x[j] = p_start.x[j]
        if not only_coords:
            cost(simplex.p[i], sample_idx)
    if not only_coords:
        simplex_sort(simplex)

    return simplex

if __name__ == '__main__':

    sample_idx = 42
    n = 3

    RHO = 1.0
    CHI = 2.0
    GAMMA = 0.5
    SIGMA = 0.5
    verbose = True

    p_r = Point(name="p_r")
    p_e = Point(name="p_e")
    p_c = Point(name="p_c")
    p_ce = Point(name="p_ce")

    p_start = Point(array([30, 620, 30]))  # start 30 620 30
    cost(p_start, sample_idx)

    simplex = initial_simplex(p_start)
    get_centroid(simplex, p_ce)

    for i in range(1, 100):
        update_point(simplex, p_ce, RHO, p_r)
        cost(p_r, sample_idx)

        if (p_r.fx < simplex.p[0].fx):
            print("difference p_r.fx < simplex.p[0].fx", f'{abs(p_r.fx - simplex.p[0].fx):.20f}')
            update_point(simplex, p_ce, RHO * CHI, p_e)
            cost(p_e, sample_idx)
            if p_e.fx < p_r.fx:
                print("difference p_e.fx < p_r.fx", f'{abs(p_e.fx - p_r.fx):.20f}')
                if verbose:
                    print("expand")
                copy_point(p_e, simplex.p[3])
            else:
                if verbose:
                    print("reflect 1")
                copy_point(p_r, simplex.p[3])
        else:
            if p_r.fx < simplex.p[2].fx:
                print("difference p_r.fx < simplex.p[2].fx", f'{abs(p_r.fx - simplex.p[2].fx):.20f}')
                if verbose:
                    print("reflect 2")
                copy_point(p_r, simplex.p[3])
            else:
                if p_r.fx < simplex.p[3].fx:
                    print("difference p_r.fx < simplex.p[3].fx", f'{abs(p_r.fx - simplex.p[3].fx):.20f}')
                    update_point(simplex, p_ce, RHO * GAMMA, p_c)
                    cost(p_c, sample_idx)
                    if p_c.fx <= p_r.fx:
                        print("difference p_c.fx <= p_r.fx", f'{abs(p_c.fx - p_r.fx):.20f}')
                        if verbose:
                            print("contract out")
                        copy_point(p_c, simplex.p[3])
                    else:
                        break  # out completely...
                else:
                    update_point(simplex, p_ce, -GAMMA, p_c)
                    cost(p_c, sample_idx)
                    if p_c.fx <= simplex.p[n].fx:
                        print("difference p_c.fx <= simplex.p[n].fx", f'{abs(p_c.fx - simplex.p[n].fx):.20f}')
                        if verbose:
                            print("contract in")
                        copy_point(p_c, simplex.p[3])
                    else:
                        break
        #print(simplex)

        #c1 = simplex.p[2].fx < simplex.p[3].fx
        #c2 = simplex.p[1].fx < simplex.p[2].fx
        #c3 = simplex.p[1].fx < simplex.p[3].fx
        #c4 = simplex.p[0].fx < simplex.p[3].fx
        #c5 = simplex.p[0].fx < simplex.p[1].fx
        #c6 = simplex.p[0].fx < simplex.p[2].fx
        #print(f"c1: {c1}, c2: {c2}, c3: {c3}, c4: {c4}, c5: {c5}, c6: {c6}")
        # insertion sort
        for k in [2, 1, 0]:
            if simplex.p[k + 1].fx < simplex.p[k].fx:
                swap_points(simplex.p[k+1], simplex.p[k])


        get_centroid(simplex, p_ce)

        print(p_r)
        print(p_e)
        print(p_c)
        print(p_ce)
        print(simplex)
        print(f"iteration {i} done\n")

    # solution in p0
    print(simplex.p[0])
