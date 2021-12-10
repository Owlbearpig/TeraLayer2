#include <stdio.h>
#include <stdlib.h>

#include "ackley.h"

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

  // cost function parameters
  ackley_param_t ackley_params;
  ackley_params.a = 20.0;
  ackley_params.b = 0.2;
  ackley_params.c = 2.0 * PI;

  // evaluate and print starting point
  printf("Initial point\n");
  ackley_fun(n, &start, &ackley_params);
  print_point(n, &start);

  // free memory
  free(start.x);

  return 0;
}
