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
def coh_tmm(pol, n_list, d_list, th_0, lam_vac):
    """
    Main "coherent transfer matrix method" calc. Given parameters of a stack,
    calculates everything you could ever want to know about how light
    propagates in it. (If performance is an issue, you can delete some of the
    calculations without affecting the rest.)

    pol is light polarization, "s" or "p".

    n_list is the list of refractive indices, in the order that the light would
    pass through them. The 0'th element of the list should be the semi-infinite
    medium from which the light enters, the last element should be the semi-
    infinite medium to which the light exits (if any exits).

    th_0 is the angle of incidence: 0 for normal, pi/2 for glancing.
    Remember, for a dissipative incoming medium (n_list[0] is not real), th_0
    should be complex so that n0 sin(th0) is real (intensity is constant as
    a function of lateral position).

    d_list is the list of layer thicknesses (front to back). Should correspond
    one-to-one with elements of n_list. First and last elements should be "inf".

    lam_vac is vacuum wavelength of the light.

    Outputs the following as a dictionary (see manual for details)

    * r--reflection amplitude

    """
    # Convert lists to numpy arrays if they're not already.
    n_list = array(n_list)
    d_list = array(d_list, dtype=float)

    # Input tests
    if ((hasattr(lam_vac, 'size') and lam_vac.size > 1)
          or (hasattr(th_0, 'size') and th_0.size > 1)):
        raise ValueError('This function is not vectorized; you need to run one '
                         'calculation at a time (1 wavelength, 1 angle, etc.)')
    if (n_list.ndim != 1) or (d_list.ndim != 1) or (n_list.size != d_list.size):
        raise ValueError("Problem with n_list or d_list!")
    assert d_list[0] == d_list[-1] == inf, 'd_list must start and end with inf!'
    assert abs((n_list[0]*np.sin(th_0)).imag) < 100*EPSILON, 'Error in n0 or th0!'
    assert is_forward_angle(n_list[0], th_0), 'Error in n0 or th0!'
    num_layers = n_list.size

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
    th_list = list_snell(n_list, th_0)

    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    kz_list = 2 * np.pi * n_list * cos(th_list) / lam_vac
    print(kz_list.real[1], kz_list.real[2])
    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)

    # For a very opaque layer, reset delta to avoid divide-by-0 and similar
    # errors. The criterion imag(delta) > 35 corresponds to single-pass
    # transmission < 1e-30 --- small enough that the exact value doesn't
    # matter.
    for i in range(1, num_layers-1):
        if delta[i].imag > 35:
            delta[i] = delta[i].real + 35j
            if 'opacity_warning' not in globals():
                global opacity_warning
                opacity_warning = True
                print("Warning: Layers that are almost perfectly opaque "
                      "are modified to be slightly transmissive, "
                      "allowing 1 photon in 10^30 to pass through. It's "
                      "for numerical stability. This warning will not "
                      "be shown again.")

    # t_list[i,j] and r_list[i,j] are transmission and reflection amplitudes,
    # respectively, coming from i, going to j. Only need to calculate this when
    # j=i+1. (2D array is overkill but helps avoid confusion.)
    t_list = zeros((num_layers, num_layers), dtype=complex)
    r_list = zeros((num_layers, num_layers), dtype=complex)
    for i in range(num_layers-1):
        t_list[i,i+1] = interface_t(pol, n_list[i], n_list[i+1],
                                    th_list[i], th_list[i+1])
        r_list[i,i+1] = interface_r(pol, n_list[i], n_list[i+1],
                                    th_list[i], th_list[i+1])

    # At the interface between the (n-1)st and nth material, let v_n be the
    # amplitude of the wave on the nth side heading forwards (away from the
    # boundary), and let w_n be the amplitude on the nth side heading backwards
    # (towards the boundary). Then (v_n,w_n) = M_n (v_{n+1},w_{n+1}). M_n is
    # M_list[n]. M_0 and M_{num_layers-1} are not defined.
    # My M is a bit different than Sernelius's, but Mtilde is the same.
    M_list = zeros((num_layers, 2, 2), dtype=complex)
    for i in range(1, num_layers-1):
        M_list[i] = (1/t_list[i,i+1]) * np.dot(
            make_2x2_array(exp(-1j*delta[i]), 0, 0, exp(1j*delta[i]),
                           dtype=complex),
            make_2x2_array(1, r_list[i,i+1], r_list[i,i+1], 1, dtype=complex))

    Mtilde = make_2x2_array(1, 0, 0, 1, dtype=complex)
    for i in range(1, num_layers-1):
        Mtilde = np.dot(Mtilde, M_list[i])
    Mtilde = np.dot(make_2x2_array(1, r_list[0,1], r_list[0,1], 1,
                                   dtype=complex)/t_list[0,1], Mtilde)

    # Net complex reflection amplitudes
    r = Mtilde[1,0]/Mtilde[0,0]

    return r

angle_in = 8 * pi / 180
freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950])

lambda_vacs = (c0 / freqs) * 10 ** -6
one = np.ones_like(freqs)

d_list = array([inf, 43.0, 641.0, 74.0, inf])

n0 = array([1.56, 1.57, 1.60, 1.61, 1.62, 1.62])
n1 = array([2.88, 2.89, 2.89, 2.90, 2.88, 2.89])
n2 = array([1.56, 1.57, 1.60, 1.61, 1.62, 1.62])

#n0 = 1.5*one
#n1 = 2.8*one
#n2 = 1.5*one

n = array([one, n0, n1, n2, one]).T

rs = []
for i, lambda_vac in enumerate(lambda_vacs):
    n_list = n[i]
    r = coh_tmm("s", n_list, d_list, angle_in, lambda_vac) * -1

    rs.append(r)
rs = array(rs)

print(rs)


