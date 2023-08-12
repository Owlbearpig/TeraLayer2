import numpy as np
from tmm import (is_forward_angle, list_snell, seterr, interface_t, interface_r,
                 make_2x2_array)
from functools import partial
from pathlib import Path
import sys
from numpy import array, pi, cos
from scipy.constants import c as c0
from consts import selected_freqs
from numfi import numfi as numfi_
from meas_eval.cw.refractive_index_fit import freqs as all_freqs
from meas_eval.cw.load_data import mean_data
from optimization.nelder_mead_nD import grid, initial_simplex, Point

EPSILON = sys.float_info.epsilon  # typical floating-point calculation error

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


# TODO structured output of a,b and f,g coeffs. Done; lgtm

# Thanks TMM package !
def sample_coefficients(pol, n, th_0, freqs):
    a, b = np.zeros_like(freqs, dtype=float), np.zeros_like(freqs, dtype=float)
    f, g = np.zeros_like(freqs, dtype=float), np.zeros_like(freqs, dtype=float)

    lambda_vacs = (c0 / freqs) * 10 ** -6
    for f_idx, lambda_vac in enumerate(lambda_vacs):
        n_list = array(n[f_idx])

        num_layers = n_list.size

        th_list = list_snell(n_list, th_0)

        kz_list = 2 * np.pi * n_list * cos(th_list) / lambda_vac
        f[f_idx], g[f_idx] = kz_list.real[1], kz_list.real[2]

        for i in range(num_layers - 1):
            fresnel_r = interface_r(pol, n_list[i], n_list[i + 1], th_list[i], th_list[i + 1])
            if i == 0:
                b[f_idx] = np.abs(fresnel_r)
            if i == 1:
                a[f_idx] = np.abs(fresnel_r)

    return a, b, f, g


def combine_module_coefficients():
    coeffs = default_coeffs()
    a, b = coeffs[0], coeffs[1]
    """
    self.c0 = self.two * self.a * (self.b * self.b - self.one)
    self.c1 = self.two * self.b
    self.c2 = self.two * self.a * (self.one + self.b * self.b)
    self.c3 = self.two * self.a * self.a * self.b
    self.c4 = self.a * self.a
    self.c5 = self.b * self.b - self.one
    self.c6 = self.b * self.b + self.one
    self.c7 = self.four * self.a * self.b
    """
    c = np.zeros((8, len(selected_freqs)))
    c[0] = 2 * a * (b * b - 1)
    c[1] = 2 * b
    c[2] = 2 * a * (1 + b * b)
    c[3] = 2 * a * a * b
    c[4] = a * a
    c[5] = b * b - 1
    c[6] = b * b + 1
    c[7] = 4 * a * b

    return c


def _verilog_code():
    print("\nVerilog assign f, g: ")
    w = pd + p
    coeffs = default_coeffs()
    fs, gs = coeffs[2], coeffs[3]

    indent = "			"

    for i, f_ in enumerate(fs):
        f_ = numfi(f_ * 2**3)
        bin_s = numfi_(i + 1, w=cntr_w, f=0).bin[0]
        if i == 0:
            print(f"assign f = (cntr == 4'b{bin_s}) ? {w}'b{f_.bin[0]}: // {f_} ({f_.w} / {f_.f})")
        else:
            print(indent + f"(cntr == 4'b{bin_s}) ? {w}'b{f_.bin[0]}: // {f_} ({f_.w} / {f_.f})")
    print(indent + "{(4+p){1'b0}};\n")

    for i, g_ in enumerate(gs):
        g_ = numfi(g_ * 2**3)
        bin_s = numfi_(i + 1, w=cntr_w, f=0).bin[0]
        if i == 0:
            print(f"assign g = (cntr == 4'b{bin_s}) ? {w}'b{g_.bin[0]}: // {g_} ({g_.w} / {g_.f})")
        else:
            print(indent + f"(cntr == 4'b{bin_s}) ? {w}'b{g_.bin[0]}: // {g_} ({g_.w} / {g_.f})")
    print(indent + "{(4+p){1'b0}};\n")

    c = combine_module_coefficients()
    print("Verilog assign c array: ")
    w = 3 + p

    for c_idx in range(8):
        for i, c_ in enumerate(c[c_idx]):
            c_ = numfi_(c_, s=1, w=3 + p, f=p, rounding="floor")
            bin_s = numfi_(i + pipe_delay, w=cntr_w, f=0).bin[0]
            if i == 0:
                print(f"assign c[{c_idx}] = (cntr == 4'b{bin_s}) ? {w}'b{c_.bin[0]}: // {c_} ({c_.w} / {c_.f})")
            else:
                print(indent + f"(cntr == 4'b{bin_s}) ? {w}'b{c_.bin[0]}: // {c_} ({c_.w} / {c_.f})")
        print(indent + "{(4+p){1'b0}};\n")


def default_coeffs():
    from meas_eval.cw.refractive_index_fit import n

    angle_in = 8 * pi / 180
    freq_idx = [np.argmin(np.abs(f - all_freqs)) for f in selected_freqs]
    one = np.ones_like(selected_freqs)

    n0, n1, n2 = np.transpose(n[freq_idx, 1:4].real)

    # n0 = array([1.513, 1.515, 1.520, 1.521, 1.522, 1.524], dtype=float)
    # n1 = array([2.782, 2.782, 2.784, 2.785, 2.786, 2.787], dtype=float)
    # n2 = array([1.513, 1.515, 1.520, 1.521, 1.522, 1.524], dtype=float)

    n = array([one, n0, n1, n2, one]).T

    return sample_coefficients("s", n, angle_in, selected_freqs)


def _sample_data(sam_idx=None):
    def _real_data_cw(s_idx):
        t_func_fd = mean_data(s_idx, ret_t_func=True)
        freq_idx_lst = []
        for freq in selected_freqs:
            f_idx = np.argmin(np.abs(freq - t_func_fd[:, 0].real))
            freq_idx_lst.append(f_idx)

        return t_func_fd[freq_idx_lst, 1]

    indent = "    "
    w = 3 + p

    def _sample_data_v(out, sam_idx_):
        r_exp = _real_data_cw(sam_idx_)
        if out is print:
            out(f"r_target: {r_exp}")
        _numfi = partial(numfi_, s=1, w=w, f=p, fixed=True, rounding="floor")
        r_exp_real = _numfi(r_exp.real)
        r_exp_imag = _numfi(r_exp.imag)

        out(f"// sam_idx = {sam_idx_}:")
        out("cur_data_real = {")
        for i in range(len(r_exp)):
            line0 = f"{w}'b{r_exp_real[i].bin[0]}"
            line1 = f" // {r_exp_real[i]} ({r_exp_real[i].w} / {r_exp_real[i].f})"
            if i == len(r_exp) - 1:
                out(indent + line0 + line1)
            else:
                out(indent + line0 + "," + line1)
        out("};\n")

        out("cur_data_imag = {")
        for i in range(len(r_exp)):
            line0 = f"{w}'b{r_exp_imag[i].bin[0]}"
            line1 = f" // {r_exp_imag[i]} ({r_exp_imag[i].w} / {r_exp_imag[i].f})"
            if i == len(r_exp) - 1:
                out(indent + line0 + line1)
            else:
                out(indent + line0 + "," + line1)
        out("};\n")

    if sam_idx is not None:
        _sample_data_v(out=print, sam_idx_=sam_idx)
    else:
        dir_ = Path("verilog_gen_output")
        dir_.mkdir(exist_ok=True)
        with open(dir_ / f"_sample_data_v.txt", "w") as file:
            def write_line(line_, indents=0):
                file.write(indents * indent + line_ + "\n")

            write_line_ = partial(write_line, indents=2)

            write_line("initial begin", 1)
            write_line("cur_data_real = {6*(3+p){1'b0}};", 2)
            write_line("cur_data_imag = {6*(3+p){1'b0}};", 2)
            write_line("#40", 2)
            _sample_data_v(out=write_line_, sam_idx_=0)
            write_line("end", 1)
            write_line("always @(posedge eval_done) begin", 1)
            write_line("eval_done_cntr <= eval_done_cntr + 1;", 2)
            write_line("case (eval_done_cntr)", 2)
            for idx in range(1, 101):
                write_line(f"{idx} : begin", 2)
                _sample_data_v(out=write_line_, sam_idx_=idx)
                write_line("end", 2)
            write_line("endcase", 2)
            write_line("end", 1)


def _sim_p(p_):
    p_ = array(p_)

    scale = (2 * pi * 2 ** input_scale)
    if all(p_ < 4.0):
        p_upscaled = p_ * scale
        p_test = p_.copy()
    else:
        p_upscaled = p_.copy()
        p_test = p_ / scale

    p_test = numfi(p_test)

    for i in range(len(p_test)):
        line = f"i_d{i}_r = {p_test.w}'b{p_test[i].bin[0]}; // {p_test[i]} ({p_test[i].w} / {p_test[i].f}) // {p_[i]}"
        line += f" {p_upscaled[i]}"
        print(line)
    print()


def _grid_point_gen():
    w_d = 4 + p
    _numfi = partial(numfi_, s=1, w=w_d, f=p, fixed=True, rounding="floor")
    c_ = 2*pi*2**input_scale
    p_center_ = _numfi(array(p_center) * (1 / c_))
    indent = "    "

    points = grid(p_center_, grid_options)

    dir_ = Path("verilog_gen_output")
    dir_.mkdir(exist_ok=True)
    with open(dir_ / f"_grid_point_gen.txt", "w") as file:
        write_line = lambda line_: file.write(indent + line_ + "\n")

        write_line("")
        write_line("initial begin")
        p0 = initial_simplex(Point(points[0]), grid_options)
        for i in range(4):
            for j in range(3):
                val = np.abs(p0.p[i].x[j])
                val_bin = numfi_(val, s=0, w=w_d, f=p, rounding="floor")

                line0 = f"p{i}_d{j}_0 = {w_d}'b{val_bin.bin[0]};"
                line1 = f"// {val}, ({val_bin.w} / {val_bin.f}) // val*scaling {array(val) * c_}"
                write_line(indent + line0 + line1)
            if i != 3:
                write_line("")
        write_line("end\n")

        p0_cnt = len(points)-1
        cntr_w = 10
        p0_cnt_bin = numfi_(p0_cnt, s=0, w=cntr_w, f=0).bin[0]
        write_line(f"reg [{cntr_w-1}:0] p0_cnt = {cntr_w}'b{p0_cnt_bin}; // total p0 cnt {p0_cnt_bin} ({p0_cnt})\n")
        write_line("always @(p0_idx) begin")
        write_line(indent + "case(p0_idx)")

        write_line("/*")
        write_line(f"total p0 count: {p0_cnt_bin} ({p0_cnt})")
        line = ""
        for i, p_ in enumerate(points[1:]):
            i += 1
            line += str((array(p_)*c_).astype(int)) + ", "
            if (i % 10) == 0:
                write_line(line)
                line = ""
        write_line(line[:-2])
        write_line("*/\n")

        for idx, point in enumerate(points[1:]):
            p0 = initial_simplex(Point(point), grid_options)

            write_line(f"// Initial point idx {idx}")
            write_line(f"{cntr_w}\'b{numfi_(idx, w=cntr_w, f=0).bin[0]} : begin")
            for i in range(4):
                for j in range(3):
                    val = np.abs(p0.p[i].x[j])
                    val_bin = numfi_(val, s=0, w=w_d, f=p, rounding="floor")

                    line0 = f"p{i}_d{j}_0 = {w_d}'b{val_bin.bin[0]};"
                    line1 = f"// {val}, ({val_bin.w} / {val_bin.f}) // val*scaling {array(val)*c_}"
                    write_line(indent + line0 + line1)
                if i != 3:
                    write_line("")
            write_line("end")
        write_line("default : begin")
        for i in range(4):
            for j in range(3):
                write_line(indent + f"p{i}_d{j}_0 = p{i}_d{j}_0;")
            if i != 3:
                write_line("")
        write_line("end")
        write_line("endcase")
        write_line("end")
        write_line("endmodule")


if __name__ == '__main__':
    np.set_printoptions(floatmode="fixed")

    #### settings #####
    cntr_w = 5  # default 5
    pd, p = 4, 11  # default 4, 11
    input_scale = 6  # default 6 probably don't change this
    pipe_delay = 5  # default 5 probably don't change this

    ### grid and simplex ####
    p_center = [100, 600, 100]  # default [150, 600, 150]
    grid_size = 2  # default 3
    grid_spacing = 28  # default 40
    grid_options = {"input_scale": input_scale, "grid_spacing": grid_spacing, "size": grid_size,
                    "simplex_spread": 40}

    numfi = partial(numfi_, s=1, w=pd + p, f=p, fixed=True, rounding="floor")

    coeffs = default_coeffs()

    from meas_eval.cw.refractive_index_fit import n

    #### output #####

    print(f"Frequencies (THz):\n{selected_freqs}\n")
    freq_idx = [np.argmin(np.abs(f - all_freqs)) for f in selected_freqs]
    print("Refractive index:\n", n[freq_idx, 1:4], "\n")

    a_s, b_s = str(coeffs[0]).replace(" ", ", "), str(coeffs[1]).replace(" ", ", ")
    f_s, g_s = str(coeffs[2]).replace(" ", ", "), str(coeffs[3]).replace(" ", ", ")

    print(f"a: {a_s}\nb: {b_s}\nf: {f_s}\ng: {g_s}")

    _verilog_code()

    # _sample_data(sam_idx=None)

    # p_sol = array([241., 661., 237.])
    # p_sol = array([43.0, 641.0, 74.0])
    # p_sol = array([146, 660, 73])
    # p_sol = array([46, 660, 73])
    # p_sol = array([42, 641, 74])
    # p_sol = array([50, 450, 100])
    # p_sol = array([50, 450, 100])
    p1 = array([0.02539062, 1.19287109, 0.17333984])
    p2 = array([0.07373047, 1.19287109, 0.17333984])
    p3 = array([0.07373047, 1.09375,    0.17333984])
    p4 = array([0.07373047, 1.19287109, 0.07421875])

    for p_ in [p1, p2, p3, p4]:
        _sim_p(p_)

    _grid_point_gen()
