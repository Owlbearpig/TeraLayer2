//
// Created by POTATO on 12/10/2021.
//
#include "loss.h"
#include <math.h>
#define PI 3.1415926535897932384626433832795

#define wl_cnt 6
//-----------------------------------------------------------------------------
// Ackley function
// - n is the dimension of the data
// - point is the location where the function will be evaluated
// - arg contains the parameters of the function
// More details on the function at http://www.sfu.ca/%7Essurjano/ackley.html
//-----------------------------------------------------------------------------

// magic numbers (they depend on geometry, wavelength and refractive index, independent of p)
const double a = 0.19737935744311108, b = 0.300922921527581;
const double f[wl_cnt] = {
13235.26131362, 16379.02884655, 20465.92663936,
25181.57793875, 26753.46170521, 29897.22923814
};

const double g[wl_cnt] = {
24705.82111877, 30574.18718023, 38203.06306014,
47005.61215233, 49939.79518306, 55808.16124453,
};

// measured reflectance (sample idx=0)
/*const double R0[wl_cnt] = {0.00935363, 0.41417285, 0.09113865,
                           0.18839357, 0.01541133, 0.08548127};
*/
// measured reflectance (sample idx=10)
const double R0[wl_cnt] = {0.01619003, 0.3079267,  0.11397636, 0.13299026, 0.05960753, 0.08666484};


void loss_fun(point_t *point) {
    double R[wl_cnt] = {0};

    for (int i=0; i < wl_cnt; i++){
        double f0 = f[i]*(point->x[0])*1e-6;
        double f1 = g[i]*(point->x[1])*1e-6;
        double f2 = f[i]*(point->x[2])*1e-6;

        double s0 = + f2 + f1 + f0;
        double s1 = +f2 - f1 - f0;
        double s2 = +f2 + f1 - f0;
        double s3 = -f2 + f1 - f0;

        double cs0 = cos(s0);
        double cs1 = cos(s1);
        double cs2 = cos(s2);
        double cs3 = cos(s3);
        double ss0 = sin(s0);
        double ss1 = sin(s1);
        double ss2 = sin(s2);
        double ss3 = sin(s3);

        double m_12_r = (1 - a * a) * b * (cs2 - cs1);
        double m_22_r = (1 - a * a) * (cs0 - b * b * cs3);

        double m_12_i = - 2 * a * (ss0 + b * b * ss3) + (a * a + 1) * b * (ss1 - ss2);
        double m_22_i = (a * a + 1) * (ss0 + b * b * ss3) + 2 * a * b * (ss2 - ss1);

        R[i] = (m_12_r * m_12_r + m_12_i * m_12_i) / (m_22_r * m_22_r + m_22_i * m_22_i);
    }

    // final result
    double total_loss = 0;
    for (int i = 0; i < wl_cnt; i++){
        total_loss += (R[i]-R0[i])*(R[i]-R0[i]);
    }
    point->fx = total_loss;
}

