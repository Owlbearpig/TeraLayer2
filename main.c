#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "loss.h"

#define PI 3.1415926535897932384626433832795

//-----------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------

int main(int argc, const char *argv[]) {

    // reading initial point from command line
    const int n = argc - 1;
    point_t start;
    start.x = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
    start.x[i] = atof(argv[i + 1]);
    }

    // evaluate and print starting point
    printf("Initial point\n");
    clock_t tic = clock();
    for (int i = 0; i < 1000; i++){
        // start.x[0] += i;
        loss_fun(&start);
    }
    clock_t toc = clock();
    printf("Elapsed: %f us\n", 1e6*((double)(toc - tic) / (1000.0*CLOCKS_PER_SEC)));

    print_point(n, &start);

    // free memory
    free(start.x);

    return 0;
}
