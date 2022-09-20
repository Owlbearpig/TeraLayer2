import matplotlib.pyplot as plt
from consts import custom_mask_420, um_to_m, THz, GHz, um
import numpy as np
from numpy import array, sum
from model.tmm import get_amplitude, get_phase, thickest_layer_approximation
from model.refractive_index import get_n
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


def initial_simplex(p_start, cost_func, sample_idx=None, fevals=0):
    simplex = Simplex(*[Point(name=f"p{i}") for i in range(n + 1)])
    for i in range(n + 1):
        for j in range(n):
            if i - 1 == j:
                if not np.isclose(p_start.x[j], 0):
                    simplex.p[i].x[j] = 0.80 * p_start.x[j]
                else:
                    simplex.p[i].x[j] = 0.00025
            else:
                simplex.p[i].x[j] = p_start.x[j]
        cost_func(simplex.p[i], sample_idx)
        fevals += 1
    simplex_sort(simplex)

    return simplex


class CostModel:
    def __init__(self, freqs, p_solution):
        self.freqs = freqs
        self.n = get_n(freqs, n_min=2.7, n_max=2.7)
        en_noise = False
        if en_noise:
            noise = (0.9 + np.random.random(len(freqs))*0.2)
        else:
            noise = 1

        self.R0_amplitude = get_amplitude(self.freqs, p_solution * um_to_m, self.n) * noise
        self.R0_phase = get_phase(freqs, p_solution * um_to_m, self.n) * noise
        self.thickest_layer = thickest_layer_approximation(freqs, self.R0_amplitude) * um

    def cost(self, point, *args):
        def cost_function(p):
            amp_loss = sum((get_amplitude(self.freqs, p, self.n) - self.R0_amplitude) ** 2)
            phase_loss = sum((get_phase(self.freqs, p, self.n) - self.R0_phase) ** 2)

            loss = np.log10(amp_loss * phase_loss)
            # loss = np.log10(amp_loss)
            # loss = amp_loss * phase_loss

            return loss

        if isinstance(point, Point):
            p = array([point.x[0], point.x[1], point.x[2]], dtype=float) * um_to_m

            point.fx = cost_function(p)
        else:
            p = point.copy() * um_to_m

            return cost_function(p)


def grid(p_center, spacing):
    size = 3
    grid_points = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                point = [p_center[0] + i * spacing, p_center[1] + j * spacing, p_center[2] + k * spacing]
                if all(point):
                    grid_points.append(point)

    return grid_points


def rand_sol():
    return [int(i) for i in [uniform(20, 300), uniform(500, 700), uniform(50, 300)]]


def is_success(sol, p):
    limit = 5
    return all([abs(sol[i]-p[i]) < limit for i in range(len(sol))])


if __name__ == '__main__':
    np.random.seed(420)
    rand = np.random.random
    from numpy.random import uniform

    all_freqs = np.arange(0.001, 1.400 + 0.001, 0.001) * THz

    test_values = [
        [325, 650, 125.], [225, 650, 125.], [125, 650, 125.], [125, 650, 375.],
        [275, 600, 175.], [325, 620, 50.], [275, 675, 200.], [400, 680, 125.], [250, 600, 250.],
        [300, 620, 200.], [200, 620, 300.], [47, 640, 74.], [90, 750, 110],
        [450, 700, 400], rand_sol(), rand_sol(), rand_sol(), rand_sol(), rand_sol()
    ]
    for _ in range(50):
        test_values.append(rand_sol())

    deviations, failures, fevals_all = [], 0, []
    with open("solutions_new.txt", "a") as file2:
        for test_value in test_values:
            p_sol = array(test_value, dtype=float)

            print("Solution: ", p_sol)
            freqs = array([0.040, 0.080, 0.150, 0.550, 0.640, 0.760]) * THz  # pretty good
            # freqs = array([0.020, 0.060, 0.150, 0.550, 0.640, 0.760]) * THz
            # freqs = all_freqs
            new_cost = CostModel(freqs, p_sol)

            cost_func = new_cost.cost

            from scipy.optimize import basinhopping

            p0 = array([150, 600, 150])
            # scipy.optimize.show_options(solver="minimize", method=None, disp=True)

            grid_spacing = 50
            # res = basinhopping(new_cost.cost, p0, niter=10, T=1, stepsize=grid_spacing,
            # minimizer_kwargs={"method": "Nelder-Mead"}, disp=True)
            # print(res)
            """
            rez = 1
            x = np.arange(0, 1000, rez)
            y1 = array([cost_func(array([d1, p_sol[1], p_sol[2]])) for d1 in x])
            y2 = array([cost_func(array([p_sol[0], d2, p_sol[2]])) for d2 in x])
            y3 = array([cost_func(array([p_sol[0], p_sol[1], d3])) for d3 in x])
            print(np.argmin(y1) * rez, np.argmin(y2) * rez, np.argmin(y3) * rez)
            plt.title(f"Loss function with solution at {p_sol}")
            plt.plot(x, y1, label=f"d1, [d, {p_sol[1]}, {p_sol[2]}]")
            plt.plot(x, y2, label=f"d2, [{p_sol[0]}, d, {p_sol[2]}]")
            plt.plot(x, y3, label=f"d3, [{p_sol[0]}, {p_sol[1]}, d]")
            plt.xticks(np.arange(0, 1000 + 100, 100))
            plt.xlabel("d ($\mu$m)")
            plt.ylabel("$log_{10}$(AmpLoss * PhaseLoss)")
            plt.legend()

            mean_val = np.mean(y1)
            y1_minimas = y1[y1 < mean_val]

            # min distance to last saddle point
            zero_passes = 0
            last_minima = np.inf
            threshold_distance = 0  # min distance between minima
            was_close0 = False
            for idx, isclose0 in enumerate(np.isclose(np.diff(y1_minimas), 0, atol=2e-7)):
                dist_last_minima = abs(idx - last_minima)
                if isclose0 * (dist_last_minima > threshold_distance) * (not was_close0):
                    zero_passes += 1
                    print(dist_last_minima)
                    last_minima = idx
                if isclose0:
                    was_close0 = True
                else:
                    was_close0 = False
            print("minima count :", zero_passes)
            """

            p0_grid = grid(p0, grid_spacing)

            fevals = 0
            global_min = [None, np.inf, None]  # x, fx, p0
            with open("solutions.txt", "w") as file:
                for start_val in p0_grid:
                    p0 = array(start_val)
                    p_start = Point(p0, name="Start point")
                    print(p_start)

                    times_shrinkd = 0
                    # RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5 # original values
                    RHO, CHI, GAMMA, SIGMA = 1.0, 2.0, 0.5, 0.5
                    # chi*rho expand,
                    verbose = False
                    save_output = True

                    p_r = Point(name="p_r")
                    p_e = Point(name="p_e")
                    p_c = Point(name="p_c")
                    p_ce = Point(name="p_ce")

                    cost_func(p_start); fevals += 1
                    simplex = initial_simplex(p_start, cost_func, fevals=fevals)
                    get_centroid(simplex, p_ce)
                    if verbose:
                        print("initial simplex and centroid:")
                        print(simplex)
                        print(p_ce, "\n")

                    iterations = 30
                    fx_vals = []
                    for h in range(0, iterations):
                        shrink = False
                        if verbose:
                            print(f"start of iteration {h}")
                        update_point(simplex, p_ce, RHO, p_r)
                        cost_func(p_r); fevals += 1

                        if p_r.fx < simplex.p[0].fx:
                            if verbose:
                                print("difference p_r.fx < simplex.p[0].fx", f'{abs(p_r.fx - simplex.p[0].fx):.20f}')
                            update_point(simplex, p_ce, RHO * CHI, p_e)
                            cost_func(p_e); fevals += 1
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
                                    cost_func(p_c); fevals += 1
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
                                    cost_func(p_c); fevals += 1
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
                                cost_func(simplex.p[i]); fevals += 1
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

                    # solution in p0 of simplex
                    print("solution (simplex.p0):", simplex.p[0])
                    if save_output:
                        file.write(f"{simplex.p[0].x}, fx={simplex.p[0].fx}, p0={p0}, "
                                   f"shrinks: {times_shrinkd}\n")

                    if simplex.p[0].fx < global_min[1]:
                        global_min = [simplex.p[0].x, simplex.p[0].fx, start_val]

                    #plt.plot(np.arange(0, iterations), array(fx_vals))
            #plt.show()
            print("Solution: ", p_sol)
            print("Best minimum: ", np.round(global_min[0], 2), global_min[1])
            success = is_success(global_min[0], p_sol)
            failures += not success
            file2.write(f"truth: {p_sol}, found: {np.round(global_min[0], 2)}, "
                        f"log(fx)={round(global_min[1], 3)}, p0: {global_min[2]}, "
                        f"success?: {success}, fevals: {fevals}\n")
            deviations.append(sum([abs(global_min[0][i] - p_sol[i]) for i in range(n)]))
            fevals_all.append(fevals)

        avg_dev_s = f"average deviation: {np.mean(array(deviations))}"
        fail_cnt_s = f"fail count: {failures}"
        func_evals_s = f"all function evals: {fevals_all}"
        print(avg_dev_s)
        print(fail_cnt_s)
        print(func_evals_s)
        print("Number of grid points: ", len(p0_grid))

        file2.write(avg_dev_s + "\n")
        file2.write(fail_cnt_s + "\n")
        file2.write(func_evals_s + "\n")
        file2.write("\n")
        # plt.show()
