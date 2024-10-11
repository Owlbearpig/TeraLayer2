import matplotlib.pyplot as plt

from model.initial_tests.explicitEvalSimple import explicit_reflectance
from consts import custom_mask_420, um_to_m, THz, GHz
from functions import format_data
import numpy as np
from numpy import array, sum
from model.tmm_reduced import get_amplitude, get_phase
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


def initial_simplex(p_start, cost_func=None, size=1.05):
    n = 3

    simplex = Simplex(Point(name="p0"), Point(name="p1"), Point(name="p2"), Point(name="p3"))
    for i in range(n + 1):
        for j in range(3):
            if i - 1 == j:
                if not np.isclose(p_start.x[j], 0):
                    # simplex.p[i].x[j] = 0.20 * p_start.x[j]
                    simplex.p[i].x[j] = size * p_start.x[j]
                else:
                    simplex.p[i].x[j] = 0.00025
            else:
                simplex.p[i].x[j] = p_start.x[j]
        if cost_func is not None:
            cost_func(simplex.p[i])

    if cost_func is not None:
        simplex_sort(simplex)

    return simplex


def cost_tmm(p, *args):
    freqs = array([0.050, 0.070, 0.150, 0.600, 0.680, 0.720]) * THz
    # freqs = array([0.050, 0.060, 0.130, 0.540, 0.680, 0.720]) * THz
    # freqs = array([0.040, 0.080, 0.160, 0.560, 0.680, 0.720]) * THz
    freqs = array([0.050, 0.090, 0.170, 0.570, 0.690, 0.730]) * THz
    freqs = array([0.040, 0.080, 0.150, 0.550, 0.640, 0.760]) * THz
    # freqs = array([0.050, 0.090, 0.170, 0.210, 0.610, 0.690]) * THz

    all_freqs = np.arange(0.001, 1.400 + 0.001, 0.001) * THz
    n = get_n(freqs, 2.70, 2.70)

    p_sol = Point(array([50, 420, 70]))
    R0_amplitude = get_amplitude(freqs, p_sol.x * um_to_m, n)
    R0_phase = get_phase(freqs, p_sol.x * um_to_m, n)

    amp_loss = sum((get_amplitude(freqs, p.x * um_to_m, n) - R0_amplitude) ** 2)
    phase_loss = sum((get_phase(freqs, p.x * um_to_m, n) - R0_phase) ** 2)

    p.fx = amp_loss * phase_loss


if __name__ == '__main__':
    # cost_func = cost_model
    from functools import partial
    from model.cost_function import Cost

    p_sol = array([193.0, 544.0, 168.0])
    p_sol = array([76., 530., 200.])
    freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950]) * THz  # GHz; freqs. set on fpga

    new_cost = Cost(freqs, p_sol, 0.00)
    cost_func = new_cost.cost # model data, noise: 0.00

    # cost_func = partial(cost_og, sample_idx=3) # real data

    times_shrinkd = 0
    # n = 3

    # RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5 # original values
    RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5
    verbose = True
    save_output = False
    en_shrinking = True
    print_state = False

    p_r = Point(name="p_r")
    p_e = Point(name="p_e")
    p_c = Point(name="p_c")
    p_ce = Point(name="p_ce")

    from random import choice
    from nelder_mead_nD import grid
    grid_pnts = grid()

    #p0 = choice(grid_pnts)
    #p0 = p_sol.copy() * 0.99
    p0 = array([150, 500, 150], dtype=float)
    p_start = Point(p0)

    cost_func(p_start)

    if print_state:
        print("b11111 simplex")
        print("b10000, b10001, b10010, b10011 calc fx of simplex")
        print("b10100 sort simplex")
    # b11111 simplex
    # b10000, b10001, b10010, b10011 calc fx of simplex
    # b10100 sort simplex
    size = 0.80
    simplex = initial_simplex(p_start, cost_func=cost_func, size=size)
    get_centroid(simplex, p_ce)  # b10101
    if print_state:
        print("b10101")

    if verbose:
        print("initial simplex and centroid:")
        print(simplex)
        print(p_ce, "\n")

    min_fx_val = []
    p0_fx_val = []
    for i in range(1, 15):
        shrink = False
        if verbose:
            print(f"start of iteration {i}")

        if print_state:
            print("b00000")
        update_point(simplex, p_ce, RHO, p_r)  # b00000
        cost_func(p_r)

        if (p_r.fx < simplex.p[0].fx):
            if verbose:
                print("difference p_r.fx < simplex.p[0].fx", f'{(p_r.fx - simplex.p[0].fx):.20f}')
            update_point(simplex, p_ce, RHO * CHI, p_e)  # b00010
            cost_func(p_e)
            if print_state:
                print("b00010")
            if p_e.fx < p_r.fx:
                if verbose:
                    print("difference p_e.fx < p_r.fx", f'{(p_e.fx - p_r.fx):.20f}')
                    print("expand")
                copy_point(p_e, simplex.p[3])  # b00100
                if print_state:
                    print("b00100")
            else:
                if verbose:
                    print("reflect 1")
                copy_point(p_r, simplex.p[3])  # b00101
                if print_state:
                    print("b00101")
        else:
            if print_state:
                print("b00011")
            if p_r.fx < simplex.p[2].fx:  # b00011
                if verbose:
                    print("difference p_r.fx < simplex.p[2].fx", f'{(p_r.fx - simplex.p[2].fx):.20f}')
                    print("reflect 2")
                copy_point(p_r, simplex.p[3])  # b00110
                if print_state:
                    print("b00110")
            else:
                if print_state:
                    print("b00111")
                if p_r.fx < simplex.p[3].fx:  # b00111
                    if verbose:
                        print("difference p_r.fx < simplex.p[3].fx",
                              f'{(p_r.fx):.4f}', f'{(simplex.p[3].fx):.4f}',
                              f'{(p_r.fx - simplex.p[3].fx):.20f}')
                    update_point(simplex, p_ce, RHO * GAMMA, p_c)  # b01000
                    cost_func(p_c)
                    if print_state:
                        print("b01000")
                    if p_c.fx <= p_r.fx:
                        if verbose:
                            print("difference p_c.fx <= p_r.fx",
                                  f'{(p_c.fx):.4f}', f'{(p_r.fx):.4f}',
                                  f'{(p_c.fx - p_r.fx):.20f}')
                            print("contract out")
                        copy_point(p_c, simplex.p[3])  # b01010
                        if print_state:
                            print("b01010")
                    else:
                        if en_shrinking:
                            shrink = True
                        else:
                            print("terminated check p_c.fx <= p_r.fx\n")
                            break  # out completely...
                else:
                    update_point(simplex, p_ce, -GAMMA, p_c)  # b01001
                    cost_func(p_c)
                    if print_state:
                        print("b01001")
                    if p_c.fx <= simplex.p[3].fx:
                        if verbose:
                            print("difference p_c.fx <= simplex.p[3].fx",
                                  f'{(p_c.fx - simplex.p[3].fx):.20f}')
                            print("contract in")
                        copy_point(p_c, simplex.p[3])  # b01100
                        if print_state:
                            print("b01100")
                    else:
                        print("shrink check p_c.fx <= simplex.p3.fx\n")
                        if en_shrinking:
                            shrink = True
                        else:
                            print("terminated check p_c.fx <= simplex.p3.fx\n")
                            break  # out completely...

        if shrink:
            # b01110 set shrunk values
            # b10110, b10111, b11000 calc fx
            if print_state:
                print("b01110 set shrunk values")
                print("b10110, b10111, b11000 calc fx")
            times_shrinkd += 1
            for i in [1, 2, 3]:
                for j in [0, 1, 2]:
                    simplex.p[i].x[j] = simplex.p[0].x[j] + SIGMA * (simplex.p[i].x[j] - simplex.p[0].x[j])
                cost_func(simplex.p[i])
            simplex_sort(simplex)
        else:
            # b10100
            if print_state:
                print("b10100")
            # insertion sort
            for k in [2, 1, 0]:
                if simplex.p[k + 1].fx < simplex.p[k].fx:
                    swap_points(simplex.p[k + 1], simplex.p[k])

        if print_state:
            print("b10101")
        get_centroid(simplex, p_ce)

        if verbose:
            print(p_r)
            print(p_e)
            print(p_c)
            print(p_ce)
            print(simplex)
            print(f"iteration {i} done\n")
        min_fx_val.append(min([simplex.p[i].fx for i in range(4)]))
        p0_fx_val.append(simplex.p[0].fx)
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
    print(f"start point:{p_start}")

    plt.title(f"initial simplex scale: {size}")
    plt.plot(min_fx_val, label="min_fx_val")
    plt.plot(p0_fx_val, label="p0_fx_val")
    plt.legend()
    plt.show()
