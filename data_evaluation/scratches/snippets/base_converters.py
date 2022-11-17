from consts import *
from functions import format_data
from optimization.nelder_mead_nD import grid, initial_simplex, Point


def bin_to_dec(bin_str, signed=True):
    # expect fixed point strings, sign_int_frac or int_frac
    splits = bin_str.split("_")
    sign = 1
    if signed:
        sign_bit, int_part, frac_part = splits
        if int(sign_bit):
            sign = -1
    else:
        int_part, frac_part = splits

    res = int(int_part, 2)
    for i, b in enumerate(frac_part):
        res += int(b) * 2 ** (-i - 1)

    return sign * res


def int_to_bin(n, int_width):
    # care if n doesnt fit in int_width
    int_part = int(n)
    bin_int = bin(int_part).replace("0b", "")
    return (int_width - len(bin_int)) * "0" + bin_int


def fraction_to_bin(frac, frac_width):
    if frac >= 1:
        frac = frac - int(frac)
    res = ""
    while frac_width > 0:
        frac *= 2

        if frac < 1:
            res += "0"
        else:
            res += "1"
            frac -= 1

        frac_width -= 1
    return res


def add_one(s):
    res = ""
    carry = True  # we add 1
    for b in s[::-1]:
        if (b == "1") and carry:
            res += "0"
            carry = True
        elif (b == "0") and carry:
            res += "1"
            carry = False
        else:
            res += b

    return res[::-1]


def dec_to_twoscompl(r, int_width=8, frac_width=23, format=False):
    if r > 0:
        res = int_to_bin(r, int_width=int_width) + "_" + fraction_to_bin(r, frac_width=frac_width)
    else:
        r = abs(r)
        res = int_to_bin(r, int_width=int_width) + "_" + fraction_to_bin(r, frac_width=frac_width)
        res = invert_bin(res)
        res = add_one(res)

    if int_width == 0:
        res = res.replace("0_", "")
    if frac_width == 0:
        res = res[:-1]
    if format:
        cntr = 0
        for char in res:
            if char in ["0", "1"]:
                cntr += 1
        return f'{cntr}\'b' + res
    else:
        return res


def real_to_bin(n, int_w=8, frac_w=23):
    # TODO fix 2s complement neg vals DONE. This is 1s complement? YES
    # default precision: W(32, 8, 23)
    if n < 0:
        sign_bit = "1"
    else:
        sign_bit = "0"

    return sign_bit + "_" + int_to_bin(n, int_width=int_w) + "_" + fraction_to_bin(n, frac_width=frac_w)


def twos_compl_to_dec(s, p=23):
    s_clean = ""
    for char in s:
        if char.isdigit():
            s_clean += char

    res = 0
    for i, b in enumerate(s_clean):
        if i == 0:
            res -= int(b) * 2 ** (len(s_clean) - 1)
        else:
            res += int(b) * 2 ** (len(s_clean) - i - 1)
    return res / (2 ** p)


def invert_bin(s):
    res = ""
    for b in s:
        if b == "0":
            res += "1"
        elif b == "1":
            res += "0"
        else:
            res += b
    return res


def unsigned_dec_to_bin(dec, int_prec=1, frac_prec=15, delimeter=True):
    if delimeter:
        return int_to_bin(dec, int_width=int_prec) + "_" + fraction_to_bin(dec, frac_width=frac_prec)
    else:
        return int_to_bin(dec, int_width=int_prec) + fraction_to_bin(dec, frac_width=frac_prec)


def sanitize_str(s):
    bad_chars = ["[", "]"]
    for char in bad_chars:
        s = s.replace(char, "")
    return s


def list_to_twos_comp(lst, pd=12, p=17):
    """ example input:
    s = "[ 36.21281372 629.51439522  52.34839101]"
    print(list_to_twos_comp(s))
    exit()
    return: ['000000100100_00110110011110101', '001001110101_10000011101011110', '000000110100_01011001001100000']
    """
    lst = sanitize_str(lst)

    ret = []
    for part in lst.split(" "):
        if any(char.isdigit() for char in part):
            ret.append(dec_to_twoscompl(float(part), pd, p))

    return ret


def print_lst_verilog(lst_str, pd=12, p=17, point_name="i"):
    """ example input:
    s = "[ 36.21281372 629.51439522  52.34839101]"
    print_lst_verilog(s)
    exit()
    """
    lst_str = lst_str.replace(",", "")
    bin_str_lst = list_to_twos_comp(lst_str, pd, p)

    for i, b_str in enumerate(bin_str_lst):
        print(f"{point_name}_d{i} = {len(b_str) - 1}'b{b_str}; // {twos_compl_to_dec(bin_str_lst[i], p)}")


def convert_measurement_to_bin(pd=3, p=23, short=False):
    short_cntr = 0
    for sample_idx in range(100):
        mask = custom_mask_420
        lam, R0 = format_data(mask=mask, sample_file_idx=sample_idx, verbose=False)
        if short and (sample_idx not in [0, 10]):
            continue
        """
        3'b000 : begin
                // sample idx 10
                cur_data <= {
                25'b000_0000010000100101000001, // 0.01619002948879064
                25'b000_0100111011010100010010, // 0.30792670455455484
                25'b000_0001110100101101100011, // 0.11397636227824881
                25'b000_0010001000001011101001, // 0.13299025931927916
                25'b000_0000111101000010011100, // 0.05960753001314391
                25'b000_0001011000101111101010 // 0.08666484218325601
                }; 
            end
        
        """
        if not short:
            print(f"8\'b{int_to_bin(sample_idx, 8)} : begin")
        else:
            print(f"3\'b{int_to_bin(short_cntr, 3)} : begin")
            short_cntr += 1
        print(f"// sample idx {sample_idx}")
        print("cur_data <= {")
        for i, R0_i in enumerate(R0):
            bin_str = f"{dec_to_twoscompl(R0_i, pd, p, format=True)}"
            #bin_str = bin_str[:21] + "0"*(22-13)
            print(bin_str + "," * (len(R0) - 1 != i) + f" // {R0_i}")
        print("};\nend")


def convert_constants_fg(pd=0, p=23):
    from model.initial_tests.explicitEvalSimple import f, g

    for i, g_ in enumerate(g):
        g_ *= um_to_m
        line_end_part = dec_to_twoscompl(g_, int_width=pd, frac_width=p, format=True)
        if i == 0:
            print(f"assign g = (cntr == 3'b{int_to_bin(i + 1, 3)}) ? {line_end_part} : // {g_} 0Q{p}")
        else:
            print(f"(cntr == 3'b{int_to_bin(i + 1, 3)}) ? {line_end_part} : // {g_} 0Q{p}")
    print("{p{1'b0}};")

    for i, f_ in enumerate(f):
        f_ *= um_to_m
        line_end_part = dec_to_twoscompl(f_, int_width=pd, frac_width=p, format=True)
        if i == 0:
            print(f"assign f = (cntr == 3'b{int_to_bin(i + 1, 3)}) ? {line_end_part} : // {f_} 0Q{p}")
        else:
            print(f"(cntr == 3'b{int_to_bin(i + 1, 3)}) ? {line_end_part} : // {f_} 0Q{p}")
    print("{p{1'b0}};")


def cordic_format_constants(pd=8, p=23):
    pi_bin_s = dec_to_twoscompl(pi, int_width=pd, frac_width=p, format=True)
    print(f"pi = {pi_bin_s}; // pi {pd}Q{p}")
    pi2_bin_s = dec_to_twoscompl(2 * pi, int_width=pd, frac_width=p, format=True)
    print(f"pi2 = {pi2_bin_s}; // 2pi {pd}Q{p}")
    pi2_inv_bin_s = dec_to_twoscompl(1 / (2 * pi), int_width=pd, frac_width=p, format=True)
    print(f"pi2_inv = {pi2_inv_bin_s}; // 1/(2pi) {pd}Q{p}")


def convert_cos_constants(pd=8, p=23):
    # print("cos approx constants")
    pi_bin_s = dec_to_twoscompl(pi, int_width=pd, frac_width=p, format=True)
    print(f"pi = {pi_bin_s}; // pi {pd}Q{p}")
    pi2_bin_s = dec_to_twoscompl(2 * pi, int_width=pd, frac_width=p, format=True)
    print(f"pi2 = {pi2_bin_s}; // 2pi {pd}Q{p}")
    pi_half = dec_to_twoscompl(pi / 2, int_width=pd, frac_width=p, format=True)
    print(f"pi_half = {pi_half}; // pi/2 {pd}Q{p}\n")
    convert_sin_constants(pd=pd, p=p)


def convert_sin_constants(pd=4, p=23):
    # print("sin approx constants")
    B = dec_to_twoscompl(4 / pi, int_width=pd, frac_width=p, format=True)
    print(f"B = {B}; // 4/pi {pd}Q{p}")
    C = dec_to_twoscompl(-4 / pi ** 2, int_width=pd, frac_width=p, format=True)
    print(f"C = {C}; // -4/pi**2 {pd}Q{p}")
    P = dec_to_twoscompl(0.225, int_width=pd, frac_width=p, format=True)
    print(f"P = {P}; // 0.225 {pd}Q{p}")
    one = dec_to_twoscompl(1, int_width=pd, frac_width=p, format=True)
    print(f"one = {one}; // 1 {pd}Q{p}")


def convert_constants_ab(pd=3, p=23):
    from model.initial_tests.explicitEvalSimple import a, b  # careful, consts.py also has a=1 defined ...

    c0 = (1 - a * a) * b
    c1 = (1 - a * a)
    c2 = (a * a + 1) * b
    c3 = - 2 * a
    c4 = (a * a + 1)
    c5 = 2 * a * b
    c6 = - 2 * a * b * b
    c7 = - (1 - a * a) * b * b
    c8 = (a * a + 1) * b * b

    cnst_lst = [c0, c1, c2, c3, c4, c5, c6, c7, c8]

    for i, cnst in enumerate(cnst_lst):
        bin_str = dec_to_twoscompl(cnst, int_width=pd, frac_width=p, format=True)
        print(f"c{i} = {bin_str}; // {cnst} {pd}Q{p}")


def convert_div_lut_constants(pd=3, p=23):
    bin_str = dec_to_twoscompl(3 / 2, int_width=pd, frac_width=p, format=True)
    print(f"reg [3+p-1:0] c1 = {bin_str}; // 3/2 {pd}Q{p}")
    bin_str = dec_to_twoscompl(3 / 4, int_width=pd, frac_width=p, format=True)
    print(f"reg [3+p-1:0] c2 = {bin_str}; // 3/4 {pd}Q{p}")


def generate_initial_simplex_and_centroid(p0, pd, p):
    from optimization.nelder_mead_fromC import Point, initial_simplex, get_centroid
    p_start = Point(p0)  # start 30 620 30

    simplex = initial_simplex(p_start, cost_func=None)

    p_ce = Point(name="p_ce")
    get_centroid(simplex, p_ce)

    for i, pnt in enumerate(simplex.p):
        print_lst_verilog(str(pnt.x), point_name=f"p{i}", pd=pd, p=p)
        print(f"p{i}_fx_o = {pd + p}'b0;\n")

    print_lst_verilog(str(np.zeros(3)), point_name=f"ce", pd=pd, p=p)
    print()


def machine_constant(pd=0, p=22):
    bin_str = dec_to_twoscompl(1 / 3, int_width=pd, frac_width=p, format=False)
    print(f"reg [p-1:0] recip_3 = {pd + p}'b{bin_str}; // 1/3 0Q{p}")


def grid_points_v(points, pd=12, p=22):
    for idx, point in enumerate(points):
        p0 = initial_simplex(Point(array(point)))

        print(f"// Initial point idx {idx}")
        print(f"8\'b{int_to_bin(idx, 8)} : begin")
        for i in range(4):
            for j in range(3):
                val_dec = p0.p[i].x[j]
                val_bin = dec_to_twoscompl(p0.p[i].x[j], pd, p, format=True)
                print(f"    p{i}_d{j}_0 = {val_bin};" + f" // {round(val_dec, 1)}")
            if i != 3:
                print()
        print("end")


# TODO make gui for the viable functions + window to paste text to translate numbers ...

if __name__ == '__main__':
    print(list_to_twos_comp("[193.0 544.0 168.0]", pd=12, p=22))
    print(twos_compl_to_dec("1011110000100100010100001", p=22))
    exit()
    p0 = array([150, 600, 150])
    grid_spacing = 50
    grid_points = grid(p0, grid_spacing)

    grid_points_v(grid_points, pd=12, p=22)

    # run these to change precision
    # machine_constant(pd=0, p=22)
    # generate_initial_simplex_and_centroid(array([30, 620, 30]), pd=12, p=22)
    s = "[31.162231 630.244385 31.162231]"
    #print_lst_verilog(s, p=22)
    #convert_measurement_to_bin(pd=3, p=22, short=False)  # pd should be 3
    # convert_constants_fg(pd=0, p=22) # pd should be 0
    # cordic_format_constants(pd=8, p=22)  # pd should be 8
    # convert_cos_constants(pd=4, p=22)  # pd should be 4
    # convert_sin_constants(pd=4, p=22) # pd should be 4
    # convert_constants_ab(pd=3, p=22)
    # convert_div_lut_constants(pd=3, p=22)
    exit()

    um_m = 10 ** -6

    f = array([13235.26131362, 16379.02884655, 20465.92663936,
               25181.57793875, 26753.46170521, 29897.22923814])
    g = array([
        24705.82111877, 30574.18718023, 38203.06306014,
        47005.61215233, 49939.79518306, 55808.16124453])

    for f_ in f:
        f_ *= um_m
        a_fp_1_8_23 = fraction_to_bin(f_, frac_width=24)
        # print(a_fp_1_8_23)

    # print()
    for g_ in g:
        g_ *= um_m
        a_fp_1_8_23 = fraction_to_bin(g_, frac_width=24)
        # print(a_fp_1_8_23)

    # exit()
    # print(bin_to_dec(a_fp_1_8_23))
    i, p = 12, 20
    n2 = dec_to_twoscompl(36.21281372, int_width=i, frac_width=p)
    n10 = twos_compl_to_dec("000000110100_01011001001100000", p=p)
    print(n2, n10)
    exit()
    # print(a_fp_1_8_23)

    # b_fp_1_8_23 = "0_" + int_to_bin(b, int_width=8) + "_" + fraction_to_bin(b, frac_width=23)
    # [0.01619003 0.3079267  0.11397636 0.13299026 0.05960753 0.08666484]
    # print(int_to_bin(15, 10))
    # print(fraction_to_bin(0.9, 6))
    # real = real_to_bin(0.764776164690310) #  0.7647761646903104
    # print(unsigned_dec_to_bin(1.07608, 1, 15))
    # print(real)
    # print(pi2_inv)
    # print(bin_to_dec(pi2_inv))
    # print(1/(2*np.pi))
    # bin_str = real_to_bin(2.928132598615621, int_w=7, frac_w=17)
    # print(bin_str)
    # dec_n = twos_compl_to_dec("00000100110101000110", p=17)
    # print(dec_n)

    # print(bin_to_dec("0_00100110_11000111000100101100101"))
    # print(twos_compl_to_dec("11111110100010010011001111010011", p=23))
    # print(twos_compl_to_dec("11111111110011000001111110100010"))
    # print(-4 / (pi * pi))
