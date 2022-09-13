import matplotlib.pyplot as plt
from consts import custom_mask_420, um_to_m, THz, GHz, um
import numpy as np
from numpy import array, sum
from model.tmm import get_amplitude, get_phase, thickest_layer_approximation
from model.refractive_index import get_n

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


def initial_simplex(p_start, cost_func, sample_idx=None):
    simplex = Simplex(*[Point(name=f"p{i}") for i in range(n + 1)])
    for i in range(n + 1):
        for j in range(n):
            if i - 1 == j:
                if not np.isclose(p_start.x[j], 0):
                    simplex.p[i].x[j] = 0.40 * p_start.x[j]
                else:
                    simplex.p[i].x[j] = 0.00025
            else:
                simplex.p[i].x[j] = p_start.x[j]
        cost_func(simplex.p[i], sample_idx)
    simplex_sort(simplex)

    return simplex


class CostModel:
    def __init__(self, freqs, p_solution):
        self.freqs = freqs
        self.n = get_n(freqs, n_min=2.7, n_max=2.7)
        self.R0_amplitude = get_amplitude(self.freqs, p_solution * um_to_m, self.n)
        self.R0_phase = get_phase(freqs, p_solution * um_to_m, self.n)
        self.thickest_layer = thickest_layer_approximation(freqs, self.R0_amplitude) * um

    def cost(self, point, *args):
        def cost_function(p):
            amp_loss = ((1 / 6) * sum((get_amplitude(self.freqs, p, self.n) - self.R0_amplitude) ** 2))
            phase_loss = ((1 / 6) * sum((get_phase(self.freqs, p, self.n) - self.R0_phase) ** 2))

            loss = np.log10(amp_loss * phase_loss)# * phase_loss

            return loss

        if isinstance(point, Point):
            p = array([point.x[0], point.x[1], point.x[2]], dtype=float) * um_to_m

            point.fx = cost_function(p)
        else:
            p = point.copy() * um_to_m

            return cost_function(p)


if __name__ == '__main__':
    np.random.seed(420)
    rand = np.random.random
    all_freqs = np.arange(0.001, 1.400 + 0.001, 0.001) * THz

    #p_sol = array([40, 630, 74], dtype=float)
    p_sol = array([365, 650, 400], dtype=float)
    print(p_sol)
    freqs = array([0.040, 0.080, 0.150, 0.550, 0.640, 0.760]) * THz
    #freqs = all_freqs
    new_cost = CostModel(freqs, p_sol)

    cost_func = new_cost.cost

    x = np.arange(0, 1000, 1)
    y1 = [cost_func(array([d1, 650, 400])) for d1 in x]
    y2 = [cost_func(array([365, d2, 400])) for d2 in x]
    y3 = [cost_func(array([365, 650, d3])) for d3 in x]
    print(np.argmin(y1), np.argmin(y2), np.argmin(y3))
    plt.scatter(x, y1, label=f"d1, {p_sol}")
    plt.scatter(x, y2, label="d2")
    plt.scatter(x, y3, label="d3")
    plt.legend()
    plt.show()

    # initial guess of free parameters
    #p0 = array([150, 481, 170])
    p0 = array([155, new_cost.thickest_layer, 350])

    p_start = Point(p0, name="Start point")
    print(p_start)
    with open("solutions.txt", "a") as file:
        for sample_idx in range(100):
            times_shrinkd = 0
            if sample_idx != 0:
                continue

            # RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5 # original values
            RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5
            verbose = False
            save_output = False

            p_r = Point(name="p_r")
            p_e = Point(name="p_e")
            p_c = Point(name="p_c")
            p_ce = Point(name="p_ce")

            cost_func(p_start, sample_idx)

            simplex = initial_simplex(p_start, cost_func)
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
                        copy_point(p_e, simplex.p[n])
                    else:
                        if verbose:
                            print("reflect 1")
                        copy_point(p_r, simplex.p[n])
                else:
                    if p_r.fx < simplex.p[n-1].fx:
                        if verbose:
                            print("difference p_r.fx < simplex.p[2].fx", f'{abs(p_r.fx - simplex.p[n-1].fx):.20f}')
                            print("reflect 2")
                        copy_point(p_r, simplex.p[n])
                    else:
                        if p_r.fx < simplex.p[n].fx:
                            if verbose:
                                print("difference p_r.fx < simplex.p[3].fx", f'{abs(p_r.fx - simplex.p[n].fx):.20f}')
                            update_point(simplex, p_ce, RHO * GAMMA, p_c)
                            cost_func(p_c, sample_idx)
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
                            cost_func(p_c, sample_idx)
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
                            simplex.p[i].x[j] = simplex.p[0].x[j] + SIGMA * (simplex.p[i].x[j] - simplex.p[0].x[j])
                        cost_func(simplex.p[i])
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
                file.write(f"{[str(simplex.p[0].x[i]) + chr(44) + chr(32) for i in range(n)]}\n")
