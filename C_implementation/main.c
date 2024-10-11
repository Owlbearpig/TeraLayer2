#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "loss.h"
#include "nelder_mead.h"

//-----------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------

int main(int argc, const char *argv[]) {

    // reading initial point from command line
    const int n = argc - 1;
    printf("n = %d\n", n);
    point_t start;
    start.x = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        start.x[i] = atof(argv[i + 1]);
    }

    // optimisation settings
    optimset_t optimset;
    optimset.tolx = 0.01;
    optimset.tolf = 0.01;
    optimset.max_iter = 1000;
    optimset.max_eval = 1000;
    optimset.verbose = 1;

    // evaluate and print starting point
    printf("Initial point\n");
    loss_fun(&start);
    print_point(n, &start);

    point_t solution;
    nelder_mead(n, &start, &solution, &loss_fun, &optimset);

    // quick timer
    int nit = 1000;
    clock_t tic = clock();
    for (int i = 0; i < nit; i++){
        loss_fun(&start);
        //nelder_mead(n, &start, &solution, &loss_fun, &optimset);
    }
    clock_t toc = clock();
    printf("Elapsed: %f us / call\n", (double) (toc - tic)*1e6 / (double) (nit*CLOCKS_PER_SEC));

    // print solution
    printf("Solution\n");
    print_point(n, &solution);

    // free memory
    free(start.x);
    free(solution.x);

    return 0;
}

