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

        double s0 = f2 + f1 + f0;
        double s1 = f2 - f1 - f0;
        double s2 = f2 + f1 - f0;
        double s3 = -f2 + f1 - f0;

        double t6_22_r = cos(s0);
        double t0_12_r = a * t6_22_r;
        double t5_12_r = -b * cos(s1);
        double t2_22_r = a * t5_12_r;
        double t1_12_r = -a * t2_22_r;
        double t2_12_r = b * cos(s2);
        double t3_22_r = -b * b * cos(s3);
        double t3_12_r = -a * t3_22_r;
        double t4_12_r = a * t3_22_r;
        double t6_12_r = -a * a * t2_12_r;
        double t7_12_r = -a * t6_22_r;

        double t0_22_r = -a * t0_12_r;
        double t1_22_r = -a * t2_12_r;
        double t4_22_r = -a * t4_12_r;
        double t5_22_r = -a * t5_12_r;
        double t7_22_r = a * t2_12_r;

        double m_12_r = t0_12_r + t1_12_r + t2_12_r + t3_12_r + t4_12_r + t5_12_r + t6_12_r + t7_12_r;
        double m_22_r = t0_22_r + t1_22_r + t2_22_r + t3_22_r + t4_22_r + t5_22_r + t6_22_r + t7_22_r;

        double t6_22_i = sin(s0);
        double t0_12_i = -a * t6_22_i;
        double t5_12_i = b * sin(s1);
        double t2_22_i = -a * t5_12_i;
        double t1_12_i = -a * t2_22_i;
        double t2_12_i = -b * sin(s2);
        double t3_22_i = b * b * sin(s3);
        double t3_12_i = -a * t3_22_i;
        double t4_12_i = -a * t3_22_i;
        double t7_22_i = -a * t2_12_i;
        double t6_12_i = -a * t7_22_i;
        double t7_12_i = -a * t6_22_i;
        double t0_22_i = a * a * t6_22_i;
        double t1_22_i = -a * t2_12_i;
        double t4_22_i = -a * t4_12_i;
        double t5_22_i = -a * t5_12_i;


        double m_12_i = t0_12_i + t1_12_i + t2_12_i + t3_12_i + t4_12_i + t5_12_i + t6_12_i + t7_12_i;
        double m_22_i = t0_22_i + t1_22_i + t2_22_i + t3_22_i + t4_22_i + t5_22_i + t6_22_i + t7_22_i;

        R[i] = (m_12_r*m_12_r + m_12_i*m_12_i)/(m_22_r*m_22_r+m_22_i*m_22_i);
    }

    // final result
    double total_loss = 0;
    for (int i = 0; i < wl_cnt; i++){
        total_loss += (R[i]-R0[i])*(R[i]-R0[i]);
    }
    point->fx = total_loss;
}

