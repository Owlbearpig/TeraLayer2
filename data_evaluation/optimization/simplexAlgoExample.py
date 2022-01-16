# simplex.py
# python 3.4.3
# demo of simplex optimization
# aka amoeba method optimization
# solves x0^2 + x1^2 + x2^2 + . . . = 0
# (the 'Sphere' function)

import random
import math  # sqrt


# ------------------------------------

def show_vector(vector):
    for i in range(len(vector)):
        if i % 8 == 0:  # 8 columns
            print("\n", end="")
        if vector[i] >= 0.0:
            print(' ', end="")
        print("%.4f" % vector[i], end="")  # 4 decimals
        print(" ", end="")
    print("\n")


# ------------------------------------

def error(position):
    # Euclidean distance to (0, 0, .. 0)
    dim = len(position)
    target = [0.0 for i in range(dim)]
    dist = 0.0
    for i in range(dim):
        dist += (position[i] - target[i]) ** 2
    return math.sqrt(dist)


# ------------------------------------

class Point:
    def __init__(self, dim, minx, maxx):
        self.position = [0.0 for i in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx - minx) *
                                random.random() + minx)

        self.error = error(self.position)  # curr error


# ------------------------------------

def Solve(dim, max_epochs, minx, maxx):
    points = [Point(dim, minx, maxx) for i in range(3)]  # 3 points

    for i in range(dim): points[0].position[i] = minx
    for i in range(dim): points[2].position[i] = maxx

    best_idx = -1
    other_idx = -1
    worst_idx = -1

    centroid = [0.0 for i in range(dim)]
    expanded = [0.0 for i in range(dim)]
    reflected = [0.0 for i in range(dim)]
    contracted = [0.0 for i in range(dim)]
    arbitrary = [0.0 for i in range(dim)]

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

        if epoch == 10:
            print("--------------------")
            print(" . . . ")

        if points[best_idx].error < 1.0e-4:
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
        for i in range(dim):
            arbitrary[i] = ((maxx - minx) * random.random() + minx)
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
    show_vector(points[best_idx].position)


# ------------------------------------

print("\nBegin simplex optimization using Python demo\n")
dim = 5
random.seed(0)

print("Goal is to solve the Sphere function in " +
      str(dim) + " variables")
print("Function has known min = 0.0 at (", end="")
for i in range(dim - 1):
    print("0, ", end="")
print("0)")

max_epochs = 1000

print("Setting max_epochs    = " + str(max_epochs))
print("\nStarting simplex algorithm\n")

Solve(dim, max_epochs, -10.0, 10.0)

print("\nSimplex algorithm complete")

print("\nEnd simplex optimization demo\n")