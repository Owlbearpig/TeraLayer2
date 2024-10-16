
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nelder_mead.h"

//-----------------------------------------------------------------------------
// Utility functions
//-----------------------------------------------------------------------------

int compare(const void *, const void *);

void simplex_sort(simplex_t *);

void get_centroid(const simplex_t *, point_t *);

int continue_minimization(const simplex_t *, int, int, const optimset_t *);

void update_point(const simplex_t *, const point_t *, double, point_t *);

//-----------------------------------------------------------------------------
// Main function
// - n is the dimension of the data
// - start is the initial point (unchanged in output)
// - solution is the minimizer
// - cost_function is a pointer to a fun_t type function
// - args are the optional arguments of cost_function
// - optimset are the optimisation settings
//-----------------------------------------------------------------------------

void nelder_mead(int n, const point_t *start, point_t *solution,
                 fun_t cost_function, const optimset_t *optimset) {

    // internal points
    point_t point_r;
    point_t point_e;
    point_t point_c;
    point_t centroid;

    // allocate memory for internal points
    point_r.x = malloc(n * sizeof(double));
    point_e.x = malloc(n * sizeof(double));
    point_c.x = malloc(n * sizeof(double));
    centroid.x = malloc(n * sizeof(double));

    int iter_count = 0;
    int eval_count = 0;

    // initial simplex has size n + 1 where n is the dimensionality of the data
    simplex_t simplex;
    simplex.n = n;
    simplex.p = malloc((n + 1) * sizeof(point_t));
    for (int i = 0; i < n + 1; i++) {
        simplex.p[i].x = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            simplex.p[i].x[j] = (i - 1 == j) ? (start->x[j] != 0.0 ? 1.05 * start->x[j] : 0.00025) : start->x[j];
        }
        cost_function(simplex.p + i);
        eval_count++;
    }
    // sort points in the simplex so that simplex.p[0] is the point having
    // minimum fx and simplex.p[n] is the one having the maximum fx
    simplex_sort(&simplex);
    // compute the simplex centroid
    get_centroid(&simplex, &centroid);
    printf("Centroid ");
    print_point(3, &centroid);
    for (int i = 0; i < n + 1; i++) {
        printf("p%d, d0: %f, d1: %f, d2: %f\n", i, simplex.p[i].x[0], simplex.p[i].x[1], simplex.p[i].x[2]);
    }
    iter_count++;

    // continue minimization until stop conditions are met
    while (continue_minimization(&simplex, eval_count, iter_count, optimset)) {
        int shrink = 0;

        if (optimset->verbose) {
            printf("Iteration %04d     ", iter_count);
        }
        update_point(&simplex, &centroid, RHO, &point_r);
        cost_function(&point_r);
        printf("simplex.p3.x: ");
        print_point(3, &simplex.p[3]);
        printf("point_r.x: ");
        print_point(3, &point_r);
        printf("point_r.fx: %f\n", point_r.fx);
        eval_count++;
        if (point_r.fx < simplex.p[0].fx) {
            update_point(&simplex, &centroid, RHO * CHI, &point_e);
            cost_function(&point_e);
            eval_count++;
            if (point_e.fx < point_r.fx) {
                // expand
                if (optimset->verbose) {
                    printf("expand          ");
                }
                copy_point(n, &point_e, simplex.p + n);
            } else {
                // reflect
                if (optimset->verbose) {
                    printf("reflect         ");
                }
                copy_point(n, &point_r, simplex.p + n);
            }
        } else {
            if (point_r.fx < simplex.p[n - 1].fx) {
                // reflect
                if (optimset->verbose) {
                    printf("reflect         ");
                }
                copy_point(n, &point_r, simplex.p + n);
            } else {
                if (point_r.fx < simplex.p[n].fx) {
                    update_point(&simplex, &centroid, RHO * GAMMA, &point_c);
                    cost_function(&point_c);
                    eval_count++;
                    if (point_c.fx <= point_r.fx) {
                        // contract outside
                        if (optimset->verbose) {
                            printf("contract out    ");
                        }
                        copy_point(n, &point_c, simplex.p + n);
                    } else {
                        // shrink
                        if (optimset->verbose) {
                            printf("shrink         ");
                        }
                        shrink = 1;
                    }
                } else {
                    update_point(&simplex, &centroid, -GAMMA, &point_c);
                    printf("point_c: ");
                    print_point(3, &point_c);
                    cost_function(&point_c);
                    eval_count++;
                    if (point_c.fx <= simplex.p[n].fx) {
                        // contract inside
                        if (optimset->verbose) {
                            printf("contract in     \n");
                        }
                        copy_point(n, &point_c, simplex.p + n);
                    } else {
                        // shrink
                        if (optimset->verbose) {
                            printf("shrink          ");
                        }
                        shrink = 1;
                    }
                }
            }
        }
        for (int i = 0; i < n + 1; i++) {
            printf("simplex b4 swap: p%d, d0: %f, d1: %f, d2: %f\n", i, simplex.p[i].x[0], simplex.p[i].x[1], simplex.p[i].x[2]);
        }
        if (shrink) {
            for (int i = 1; i < n + 1; i++) {
                for (int j = 0; j < n; j++) {
                    simplex.p[i].x[j] = simplex.p[0].x[j] + SIGMA * (simplex.p[i].x[j] - simplex.p[0].x[j]);
                }
                cost_function(simplex.p + i);
                eval_count++;
            }
            simplex_sort(&simplex);
        } else {
            for (int i = n - 1; i >= 0 && simplex.p[i + 1].fx < simplex.p[i].fx; i--) {
                swap_points(simplex.p + (i + 1), simplex.p + i);
            }
        }
        get_centroid(&simplex, &centroid);
        iter_count++;
        printf("Centroid: ");
        print_point(3, &centroid);
        for (int i = 0; i < n + 1; i++) {
            printf("p%d, d0: %f, d1: %f, d2: %f\n", i, simplex.p[i].x[0], simplex.p[i].x[1], simplex.p[i].x[2]);
        }


        if (optimset->verbose) {
            // print current minimum
            printf("[ ");
            for (int i = 0; i < n; i++) {
                printf("%.2f ", simplex.p[0].x[i]);
            }
            printf("]    %.2f \n", simplex.p[0].fx);
        }
    }


    // save solution in output argument
    solution->x = malloc(n * sizeof(double));
    copy_point(n, simplex.p + 0, solution);

    // free memory
    free(centroid.x);
    free(point_r.x);
    free(point_e.x);
    free(point_c.x);
    for (int i = 0; i < n + 1; i++) {
        free(simplex.p[i].x);
    }
    free(simplex.p);
}

//-----------------------------------------------------------------------------
// Simplex sorting
//-----------------------------------------------------------------------------

int compare(const void *arg1, const void *arg2) {
    const double fx1 = ((const point_t *)arg1)->fx;
    const double fx2 = ((const point_t *)arg2)->fx;
    return (fx1 > fx2) - (fx1 < fx2);
}

void simplex_sort(simplex_t *simplex) {
    qsort((void *)(simplex->p), simplex->n + 1, sizeof(point_t), compare);
}

//-----------------------------------------------------------------------------
// Get centroid (average position) of simplex
//-----------------------------------------------------------------------------

void get_centroid(const simplex_t *simplex, point_t *centroid) {
    for (int j = 0; j < simplex->n; j++) {
        centroid->x[j] = 0;
        for (int i = 0; i < simplex->n; i++) {
            centroid->x[j] += simplex->p[i].x[j];
        }
        centroid->x[j] /= simplex->n;
    }
}

//-----------------------------------------------------------------------------
// Asses if simplex satisfies the minimization requirements
//-----------------------------------------------------------------------------

int continue_minimization(const simplex_t *simplex, int eval_count,
                          int iter_count, const optimset_t *optimset) {
    if (eval_count > optimset->max_eval || iter_count > optimset->max_iter) {
    // stop if #evals or #iters are greater than the max allowed
        return 0;
    }
    // check fx tolerance condition on fx - input simplex is assumed to be sorted
    const int n = simplex->n;
    const double condf = simplex->p[n].fx - simplex->p[0].fx;

    // check fx tolerance condition on x
    double condx = -1.0;
    for (int i = 1; i < n + 1; i++) {
        for (int j = 0; j < n; j++) {
            const double temp = fabs(simplex->p[0].x[j] - simplex->p[i].x[j]);
            if (condx < temp) {
                condx = temp;
            }
        }
    }
    // continue if both tolx or tolf condition is not met
    if (optimset->verbose) {
        printf("nit: %d, nfev: %d\n", iter_count, eval_count);
        printf("condx: %.20f, condf: %.20f\n\n", condx, condf);
    }
    return condx > optimset->tolx || condf > optimset->tolf;
}

//-----------------------------------------------------------------------------
// Update current point
//-----------------------------------------------------------------------------

void update_point(const simplex_t *simplex, const point_t *centroid,
                  double lambda, point_t *point) {
const int n = simplex->n;
    for (int j = 0; j < n; j++) {
        point->x[j] = (1.0 + lambda) * centroid->x[j] - lambda * simplex->p[n].x[j];
    }
}
