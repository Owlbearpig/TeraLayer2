//
// Created by POTATO on 12/10/2021.
//
#include "loss.h"
#include <math.h>

#define wl_cnt 6
//-----------------------------------------------------------------------------
// Ackley function
// - n is the dimension of the data
// - point is the location where the function will be evaluated
// - arg contains the parameters of the function
// More details on the function at http://www.sfu.ca/%7Essurjano/ackley.html
//-----------------------------------------------------------------------------

// magic numbers (they depend on geometry, wavelength and refractive index, independent of p)
const double complex a = 0.19737935744311108, b = 0.300922921527581;
const double complex f[wl_cnt] = {
13235.26131362*I, 16379.02884655*I, 20465.92663936*I,
25181.57793875*I, 26753.46170521*I, 29897.22923814*I
};

const double complex g[wl_cnt] = {
24705.82111877*I, 30574.18718023*I, 38203.06306014*I,
47005.61215233*I, 49939.79518306*I, 55808.16124453*I,
};

// measured reflectance (sample idx=0)
/*const double R0[wl_cnt] = {0.00935363, 0.41417285, 0.09113865,
                           0.18839357, 0.01541133, 0.08548127};
*/
// measured reflectance (sample idx=10)
const double R0[wl_cnt] = {0.01340308, 0.30089288, 0.08711644,
                           0.12522516, 0.07773267, 0.16697741};


void loss_fun(point_t *point) {
    double R[wl_cnt] = {0};

    for (int i=0; i < wl_cnt; i++){
        double complex f0 = f[i]*(point->x[0])*1e-6;
        double complex f1 = g[i]*(point->x[1])*1e-6;
        double complex f2 = f[i]*(point->x[2])*1e-6;

        double complex t0_12 = a * cexp(-f2-f1-f0);
        double complex t1_12 = b * a * a * cexp(f2-f1-f0);
        double complex t2_12 = b * cexp(-f2-f1+f0);
        double complex t3_12 = b * a * b * cexp(f2-f1+f0);
        double complex t4_12 = -b * a * b * cexp(-f0-f2+f1);
        double complex t5_12 = -b * cexp(-f2+f1+f0);
        double complex t6_12 = -b * a * a * cexp(-f0+f2+f1);
        double complex t7_12 = -a * cexp(f2+f1+f0);

        double complex t0_22 = -a * a * cexp(-f1-f0-f2);
        double complex t1_22 = -b * a * cexp(-f1+f0-f2);
        double complex t2_22 = -b * a * cexp(f2-f0-f1);
        double complex t3_22 = -b * b * cexp(f2-f1+f0);
        double complex t4_22 = b * a * b * a * cexp(-f0-f2+f1);
        double complex t5_22 = a * b * cexp(-f2+f0+f1);
        double complex t6_22 = cexp(f2+f1+f0);
        double complex t7_22 = b * a * cexp(f2+f1-f0);

        double complex m_12 = t0_12 + t1_12 + t2_12 + t3_12 + t4_12 + t5_12 + t6_12 + t7_12;
        double complex m_22 = t0_22 + t1_22 + t2_22 + t3_22 + t4_22 + t5_22 + t6_22 + t7_22;

        double complex r = m_12 / m_22;

        R[i] = creal(r) * creal(r) + cimag(r) * cimag(r);

    }

    // final result
    double total_loss = 0;
    for (int i = 0; i < wl_cnt; i++){
        total_loss += (R[i]-R0[i])*(R[i]-R0[i]);
    }
    point->fx = total_loss;
}

