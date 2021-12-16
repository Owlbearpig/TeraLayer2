//
// Created by POTATO on 12/10/2021.
//
#include "loss.h"
#include <math.h>

const float PI = (float) 3.1415926535;
const float PI_2 = (float) 6.283185307;
const float PI_half = (float) 1.57079632675;

#define wl_cnt 6

// magic numbers (they depend on geometry, wavelength and refractive index, independent of p)
const float a = (float) 0.19737935744311108, b = (float) 0.300922921527581;
const float f[wl_cnt] = {
        (float) 13235.26131362, (float) 16379.02884655, (float) 20465.92663936,
        (float) 25181.57793875, (float) 26753.46170521, (float) 29897.22923814
};

const float g[wl_cnt] = {
        (float) 24705.82111877, (float) 30574.18718023, (float) 38203.06306014,
        (float) 47005.61215233, (float) 49939.79518306, (float) 55808.16124453,
};

// measured reflectance (sample idx=0)
/*const float R0[wl_cnt] = {0.00935363, 0.41417285, 0.09113865,
                           0.18839357, 0.01541133, 0.08548127};
*/
// measured reflectance (sample idx=10)
const float R0[wl_cnt] = {(float) 0.01619003, (float) 0.3079267,  (float) 0.11397636,
                          (float) 0.13299026, (float) 0.05960753, (float) 0.08666484};

float sine(float x) {
    const float B = 4/PI;
    const float C = -4/(PI*PI);

    float y = B * x + C * x * fabsf(x);
    const float P = (float) 0.225;

    return P * (y * fabsf(y) - y) + y;   // Q * y + P * y * abs(y)
}

float cose(float x) {
    x += PI_half;
    x -= (float) (x > PI) * PI_2;

    return sine(x);
}


void loss_fun(point_t *point) {
    float R[wl_cnt] = {0};

    for (int i=0; i < wl_cnt; i++){
        float f0 = f[i]*(point->x[0])*(float) 1e-6;
        float f1 = g[i]*(point->x[1])*(float) 1e-6;
        float f2 = f[i]*(point->x[2])*(float) 1e-6;

        float s0 = +f2 + f1 + f0;
        float s1 = +f2 - f1 - f0;
        float s2 = +f2 + f1 - f0;
        float s3 = -f2 + f1 - f0;

        // map input to -pi, pi (x (python)mod 2pi - pi)
        s0 -= (PI_2 * (float) ((int) (s0/PI_2)-(s0 < 0)) + PI);
        s1 -= (PI_2 * (float) ((int) (s1/PI_2)-(s1 < 0)) + PI);
        s2 -= (PI_2 * (float) ((int) (s2/PI_2)-(s2 < 0)) + PI);
        s3 -= (PI_2 * (float) ((int) (s3/PI_2)-(s3 < 0)) + PI);

        float cs0 = cose(s0);
        float cs1 = cose(s1);
        float cs2 = cose(s2);
        float cs3 = cose(s3);
        float ss0 = sine(s0);
        float ss1 = sine(s1);
        float ss2 = sine(s2);
        float ss3 = sine(s3);

        // this part downwards takes 0.5 us
        float m_12_r = (1 - a * a) * b * (cs2 - cs1);
        float m_22_r = (1 - a * a) * (cs0 - b * b * cs3);

        float m_12_i = - 2 * a * (ss0 + b * b * ss3) + (a * a + 1) * b * (ss1 - ss2);
        float m_22_i = (a * a + 1) * (ss0 + b * b * ss3) + 2 * a * b * (ss2 - ss1);

        R[i] = (m_12_r * m_12_r + m_12_i * m_12_i) / (m_22_r * m_22_r + m_22_i * m_22_i);
    }

    // final result
    float total_loss = 0;
    for (int i = 0; i < wl_cnt; i++){
        total_loss += (R[i]-R0[i])*(R[i]-R0[i]);
    }
    point->fx = total_loss;
}


