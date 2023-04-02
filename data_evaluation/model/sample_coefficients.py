import numpy as np
from tmm import (is_forward_angle, list_snell, seterr, interface_t, interface_r,
                make_2x2_array)
import sys
from consts import *
EPSILON = sys.float_info.epsilon # typical floating-point calculation error

""" notes
1. snell(n, th)
2. phases(snell, n, lam)
3. delta = phases * d
4. t_list(n, snell) (interfaces)
5. exp(delta)
6. M = (Pi * Ri)*(Pi * Ri)*(Pi * Ri)
7. r = M(1,0) / M(0,0)

1. snell(n, th)
2. phases(snell, n, lam)
3. t_list(n, snell)

4. delta = phases * d
5. Pi = exp(-1j*delta[i]), 0, 0, exp(1j*delta[i])
6. M = (Pi * Ri)*(Pi * Ri)*(Pi * Ri)
7. r = M(1,0) / M(0,0)

d_list
[inf, 43.0, 641.0, 74.0, inf]

kz_list
[0.00871688 0.01367722 0.02532172 0.01367722 0.00871688]
[0.01079233 0.01704312 0.03145982 0.01704312 0.01079233]
[0.01349041 0.02171417 0.03932477 0.02171417 0.01349041]
[0.01660359 0.02689344 0.04856758 0.02689344 0.01660359]
[0.01764131 0.02875309 0.05124635 0.02875309 0.01764131]
[0.01971676 0.03213581 0.05747466 0.03213581 0.01971676]

delta = kz_list * d
[        inf  0.58812049 16.23122493  1.01211433         inf]
[        inf  0.73285413 20.16574242  1.26119082         inf]
[        inf  0.93370946 25.20717803  1.60684885         inf]
[        inf  1.15641785 31.13181845  1.99011443         inf]
[        inf  1.23638288 32.84890759  2.12772867         inf]
[        inf  1.38183969 36.84126019  2.37804969         inf]

Ri
[[ 0.         -0.22150194  0.          0.          0.        ]
 [ 0.          0.         -0.29858508  0.          0.        ]
 [ 0.          0.          0.          0.29858508  0.        ]
 [ 0.          0.          0.          0.          0.22150194]
 [ 0.          0.          0.          0.          0.        ]]
[[ 0.         -0.22456211  0.          0.          0.        ]
 [ 0.          0.         -0.2972335   0.          0.        ]
 [ 0.          0.          0.          0.2972335   0.        ]
 [ 0.          0.          0.          0.          0.22456211]
 [ 0.          0.          0.          0.          0.        ]]
[[ 0.         -0.23359907  0.          0.          0.        ]
 [ 0.          0.         -0.28851412  0.          0.        ]
 [ 0.          0.          0.          0.28851412  0.        ]
 [ 0.          0.          0.          0.          0.23359907]
 [ 0.          0.          0.          0.          0.        ]]
[[ 0.         -0.23656448  0.          0.          0.        ]
 [ 0.          0.         -0.28722302  0.          0.        ]
 [ 0.          0.          0.          0.28722302  0.        ]
 [ 0.          0.          0.          0.          0.23656448]
 [ 0.          0.          0.          0.          0.        ]]
[[ 0.         -0.23950689  0.          0.          0.        ]
 [ 0.          0.         -0.28116768  0.          0.        ]
 [ 0.          0.          0.          0.28116768  0.        ]
 [ 0.          0.          0.          0.          0.23950689]
 [ 0.          0.          0.          0.          0.        ]]
[[ 0.         -0.23950689  0.          0.          0.        ]
 [ 0.          0.         -0.28276671  0.          0.        ]
 [ 0.          0.          0.          0.28276671  0.        ]
 [ 0.          0.          0.          0.          0.23950689]
 [ 0.          0.          0.          0.          0.        ]]
 
0.22150193517531785 0.2985850843731809
0.22456210755125822 0.29723349994112047
0.23359906762293864 0.2885141169419409
0.23656447616946638 0.2872230174589835
0.2395068881899639 0.2811676766529977
0.2395068881899639 0.2827667100113409

0.013677220648274348 0.025321723752706075
0.017043119211226317 0.031459816571280115
0.021714173597089076 0.03932477071410015
0.026893438266979167 0.0485675794867236
0.02875309017444406 0.051246345690000396
0.03213580666555513 0.05747466488983867

"""

# TODO structured output of a,b and f,g coeffs.

# Thanks TMM package !
def sample_coefficients(pol, n, th_0, freqs):
    a, b = np.zeros_like(freqs), np.zeros_like(freqs)
    f, g = np.zeros_like(freqs), np.zeros_like(freqs)

    lambda_vacs = (c0 / freqs) * 10 ** -6
    for f_idx, lambda_vac in enumerate(lambda_vacs):
        n_list = array(n[f_idx])

        num_layers = n_list.size

        th_list = list_snell(n_list, th_0)

        kz_list = 2 * np.pi * n_list * cos(th_list) / lambda_vac
        f[f_idx], g[f_idx] = kz_list.real[1], kz_list.real[2]

        for i in range(num_layers-1):
            fresnel_r = interface_r(pol, n_list[i], n_list[i+1], th_list[i], th_list[i+1])
            if i == 0:
                b[f_idx] = np.abs(fresnel_r)
            if i == 1:
                a[f_idx] = np.abs(fresnel_r)

    return a, b, f, g


angle_in = 8 * pi / 180
freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950])

one = np.ones_like(freqs)

n0 = array([1.56, 1.57, 1.60, 1.61, 1.62, 1.62])
n1 = array([2.88, 2.89, 2.89, 2.90, 2.88, 2.89])
n2 = array([1.56, 1.57, 1.60, 1.61, 1.62, 1.62])

n0 = 1.5*one
n1 = 2.8*one
n2 = 1.5*one

n = array([one, n0, n1, n2, one]).T

coeffs = sample_coefficients("s", n, angle_in, freqs)

print(f"a: {coeffs[0]}\nb: {coeffs[1]}\nf: {coeffs[2]}\ng: {coeffs[3]}")
