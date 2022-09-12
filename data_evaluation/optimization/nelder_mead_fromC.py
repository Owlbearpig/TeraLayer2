from model.initial_tests.explicitEvalSimple import explicit_reflectance
from consts import custom_mask_420, um_to_m
from functions import format_data
import numpy as np
from numpy import array, sum
from model.tmm import get_amplitude, get_phase
from model.refractive_index import get_n


class Simplex:
    def __init__(self, p0=None, p1=None, p2=None, p3=None):
        self.p = [p0, p1, p2, p3]

    def __repr__(self):
        s = ""
        for i, p in enumerate(self.p):
            s += str(p) + "\n" * (i != 3)
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


def cost_og(p, sample_idx=10):
    mask = custom_mask_420

    _, R0 = format_data(mask=mask, sample_file_idx=sample_idx, verbose=False)

    p.fx = sum((explicit_reflectance(p.x * um_to_m) - R0) ** 2)


def cost_model(p, *args):
    p_sol = Point(array([50, 500, 50]))

    R0 = explicit_reflectance(p_sol.x * um_to_m)

    p.fx = sum((explicit_reflectance(p.x * um_to_m) - R0) ** 2)


def simplex_sort(simplex):
    simplex.p = sorted(simplex.p, key=lambda p: p.fx)
    for i, p in enumerate(simplex.p):
        p.name = f"p{i}"


def initial_simplex(p_start, only_coords=False, cost_func=cost_og):
    n = 3

    simplex = Simplex(Point(name="p0"), Point(name="p1"), Point(name="p2"), Point(name="p3"))
    for i in range(n + 1):
        for j in range(3):
            if i - 1 == j:
                if not np.isclose(p_start.x[j], 0):
                    simplex.p[i].x[j] = 0.10 * p_start.x[j]
                else:
                    simplex.p[i].x[j] = 0.00025
            else:
                simplex.p[i].x[j] = p_start.x[j]
        if not only_coords:
            cost_func(simplex.p[i], sample_idx)
    if not only_coords:
        simplex_sort(simplex)

    return simplex



def cost_tmm(p, *args):
    freqs = array([0.050, 0.070, 0.150, 0.600, 0.680, 0.720]) * 10 ** 12
    n = get_n(freqs, 2.70, 2.70)

    p_sol = Point(array([50, 500, 70]))
    R0_amplitude = get_amplitude(freqs, p_sol.x * um_to_m, n)
    R0_phase = get_phase(freqs, p_sol.x * um_to_m, n)

    amp_loss = sum((get_amplitude(freqs, p.x * um_to_m, n) - R0_amplitude) ** 2)
    phase_loss = sum((get_phase(freqs, p.x * um_to_m, n) - R0_phase) ** 2)

    p.fx = amp_loss * phase_loss


if __name__ == '__main__':
    from model.tmm import thickest_layer_approximation
    # cost_func = cost_model

    cost_func = cost_tmm

    with open("solutions.txt", "a") as file:
        for sample_idx in range(100):
            times_shrinkd = 0
            if sample_idx != 0:
                continue
            print(sample_idx)
            #n = 3

            # RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5 # original values
            RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5
            verbose = True
            save_output = False

            p_r = Point(name="p_r")
            p_e = Point(name="p_e")
            p_c = Point(name="p_c")
            p_ce = Point(name="p_ce")

            freqs = array([0.050, 0.070, 0.150, 0.600, 0.680, 0.720]) * 10 ** 12
            n = get_n(freqs, 2.70, 2.70)

            p_sol = Point(array([50, 500, 70]))
            R0_amplitude = get_amplitude(freqs, p_sol.x * um_to_m, n)

            p0 = array([150, 100, 150])
            p0[1] = thickest_layer_approximation(freqs, R0_amplitude) * 10**6

            p_start = Point(p0)  # start 30 620 30

            cost_func(p_start, sample_idx)

            simplex = initial_simplex(p_start, cost_func=cost_func)
            get_centroid(simplex, p_ce)
            if verbose:
                print("initial simplex and centroid:")
                print(simplex)
                print(p_ce, "\n")

            for i in range(1, 200):
                shrink = False
                if verbose:
                    print(f"start of iteration {i}")
                update_point(simplex, p_ce, RHO, p_r)
                cost_func(p_r, sample_idx)

                if (p_r.fx < simplex.p[0].fx):
                    if verbose:
                        print("difference p_r.fx < simplex.p[0].fx", f'{abs(p_r.fx - simplex.p[0].fx):.20f}')
                    update_point(simplex, p_ce, RHO * CHI, p_e)
                    cost_func(p_e, sample_idx)
                    if p_e.fx < p_r.fx:
                        if verbose:
                            print("difference p_e.fx < p_r.fx", f'{abs(p_e.fx - p_r.fx):.20f}')
                            print("expand")
                        copy_point(p_e, simplex.p[3])
                    else:
                        if verbose:
                            print("reflect 1")
                        copy_point(p_r, simplex.p[3])
                else:
                    if p_r.fx < simplex.p[2].fx:
                        if verbose:
                            print("difference p_r.fx < simplex.p[2].fx", f'{abs(p_r.fx - simplex.p[2].fx):.20f}')
                            print("reflect 2")
                        copy_point(p_r, simplex.p[3])
                    else:
                        if p_r.fx < simplex.p[3].fx:
                            if verbose:
                                print("difference p_r.fx < simplex.p[3].fx", f'{abs(p_r.fx - simplex.p[3].fx):.20f}')
                            update_point(simplex, p_ce, RHO * GAMMA, p_c)
                            cost_func(p_c, sample_idx)
                            if p_c.fx <= p_r.fx:
                                if verbose:
                                    print("difference p_c.fx <= p_r.fx", f'{abs(p_c.fx - p_r.fx):.20f}')
                                    print("contract out")
                                copy_point(p_c, simplex.p[3])
                            else:
                                print("shrink check p_c.fx <= p_r.fx\n")
                                shrink = True
                                # print("terminated check p_c.fx <= p_r.fx\n")
                                # break  # out completely...
                        else:
                            update_point(simplex, p_ce, -GAMMA, p_c)
                            cost_func(p_c, sample_idx)
                            if p_c.fx <= simplex.p[3].fx:
                                if verbose:
                                    print("difference p_c.fx <= simplex.p[3].fx",
                                          f'{abs(p_c.fx - simplex.p[3].fx):.20f}')
                                    print("contract in")
                                copy_point(p_c, simplex.p[3])
                            else:
                                print("shrink check p_c.fx <= simplex.p3.fx\n")
                                shrink = True
                                # print("terminated check p_c.fx <= simplex.p3.fx\n")
                                # break
                if shrink:
                    times_shrinkd += 1
                    for i in [1, 2, 3]:
                        for j in [0, 1, 2]:
                            simplex.p[i].x[j] = simplex.p[0].x[j] + SIGMA * (simplex.p[i].x[j] - simplex.p[0].x[j])
                        cost_func(simplex.p[i])
                    simplex_sort(simplex)
                else:
                    # insertion sort
                    for k in [2, 1, 0]:
                        if simplex.p[k + 1].fx < simplex.p[k].fx:
                            swap_points(simplex.p[k + 1], simplex.p[k])

                get_centroid(simplex, p_ce)
                if verbose:
                    print(p_r)
                    print(p_e)
                    print(p_c)
                    print(p_ce)
                    print(simplex)
                    print(f"iteration {i} done\n")

            if verbose:
                print(f"times shrinked: {times_shrinkd}")
                print("final point values: ")
                print(p_r)
                print(p_e)
                print(p_c)
                print(p_ce)
                print(simplex, "\n")

            # solution in p0 of simplex
            print("solution (simplex.p0):", simplex.p[0])
            if save_output:
                file.write(f"[{simplex.p[0].x[0], simplex.p[0].x[1], simplex.p[0].x[2]} ]\n")
