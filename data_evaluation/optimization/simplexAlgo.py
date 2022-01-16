import random
import numpy as np
from consts import um_to_m, um
from model.explicitEvalOptimized import explicit_reflectance
from numpy import array
from model.explicitEvalOptimized import R0 as R


def error(p):
    p = array(p)
    return sum((explicit_reflectance(p).real - R.real) ** 2)


class Point:
    def __init__(self, dim, lb, ub):
        self.position = array([0.0]*dim)

        self.position = ((ub - lb) * np.random.random(dim) + lb)

        self.error = error(self.position)  # curr error


def Solve(dim, max_epochs, lb, ub):
    points = [Point(dim, lb, ub) for i in range(3)]  # 3 points

    points[0].position = lb
    points[-1].position = ub

    best_idx = -1
    other_idx = -1
    worst_idx = -1

    centroid = [0.0]*dim
    expanded = [0.0]*dim
    reflected = [0.0]*dim
    contracted = [0.0]*dim
    arbitrary = [0.0]*dim

    epoch = 0
    while epoch < max_epochs:
        epoch += 1

        # identify best, other, worst
        if (points[0].error < points[1].error and
                points[0].error < points[2].error):
            if points[1].error < points[2].error:
                best_idx = 0
                other_idx = 1
                worst_idx = 2
            else:
                best_idx = 0
                other_idx = 2
                worst_idx = 1
        elif (points[1].error < points[0].error and
              points[1].error < points[2].error):
            if points[0].error < points[2].error:
                best_idx = 1
                other_idx = 0
                worst_idx = 2
            else:
                best_idx = 1
                other_idx = 2
                worst_idx = 0
        else:
            if points[0].error < points[1].error:
                best_idx = 2
                other_idx = 0
                worst_idx = 1
            else:
                best_idx = 2
                other_idx = 1
                worst_idx = 0

        if epoch <= 9 or epoch >= 30:
            print("--------------------")
            print("epoch = " + str(epoch) + " ", end="")
            print("best error = ", end="")
            print("%.6f" % points[best_idx].error, end="")

        if points[best_idx].error < 0.075:
            if epoch <= 9 or epoch >= 30:
                print(" reached small error. halting")
            break

        # make the centroid
        for i in range(dim):
            centroid[i] = (points[other_idx].position[i] +
                           points[best_idx].position[i]) / 2.0

        # try the expanded point
        for i in range(dim):
            expanded[i] = centroid[i] + (2.0 * (centroid[i] -
                                                points[worst_idx].position[i]))
        expanded_err = error(expanded)
        if expanded_err < points[worst_idx].error:
            if epoch <= 9 or epoch >= 30:
                print(" expanded found better error than worst error")
            for i in range(dim):
                points[worst_idx].position[i] = expanded[i]
            points[worst_idx].error = expanded_err
            continue

        # try the reflected point
        for i in range(dim):
            reflected[i] = centroid[i] + (1.0 * (centroid[i] -
                                                 points[worst_idx].position[i]))
        reflected_err = error(reflected)
        if reflected_err < points[worst_idx].error:
            if epoch <= 9 or epoch >= 30:
                print(" reflected found better error than worst error")
            for i in range(dim):
                points[worst_idx].position[i] = reflected[i]
            points[worst_idx].error = reflected_err
            continue

        # try the contracted point
        for i in range(dim):
            contracted[i] = centroid[i] + (-0.5 * (centroid[i] -
                                                   points[worst_idx].position[i]))
        contracted_err = error(contracted)
        if contracted_err < points[worst_idx].error:
            if epoch <= 9 or epoch >= 30:
                print(" contracted found better error than worst error")
            for i in range(dim):
                points[worst_idx].position[i] = contracted[i]
            points[worst_idx].error = contracted_err
            continue

        # try a random point
        arbitrary = ((ub - lb) * np.random.random(dim) + lb)
        arbitrary_err = error(arbitrary)
        if arbitrary_err < points[worst_idx].error:
            if epoch <= 9 or epoch >= 30:
                print(" arbitrary found better error than worst error")
            for i in range(dim):
                points[worst_idx].position[i] = arbitrary[i]
            points[worst_idx].error = arbitrary_err
            continue

        # could not find better point so shrink worst and other
        if epoch <= 9 or epoch >= 30:
            print(" shrinking")
        # 1. worst -> best
        for i in range(dim):
            points[worst_idx].position[i] = (points[worst_idx].position[i]
                                             + points[best_idx].position[i]) / 2.0
        points[worst_idx].error = error(points[worst_idx].position)

        # 2. other -> best
        for i in range(dim):
            points[other_idx].position[i] = (points[other_idx].position[i]
                                             + points[best_idx].position[i]) / 2.0
        points[other_idx].error = error(points[other_idx].position)

    # end-while

    print("--------------------")
    print("\nBest position found=")
    print(points[best_idx].position * um)
    print(points[best_idx].error)


# ------------------------------------

print("\nBegin simplex optimization using Python demo\n")
dim = 3
random.seed(0)
max_epochs = 1000
d0 = array([40, 650, 40]) * um_to_m

lb = d0 - array([10, 25, 10]) * um_to_m
ub = d0 + array([10, 25, 10]) * um_to_m

Solve(dim, max_epochs, lb, ub)
