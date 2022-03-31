import numpy as np
from numpy import array, pi

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
        res += int(b)*2**(-i-1)

    return sign*res


def int_to_bin(n, int_width):
    # care if n doesnt fit in int_width
    int_part = int(n)
    bin_int = bin(int_part).replace("0b", "")
    return (int_width - len(bin_int)) * "0" + bin_int


def fraction_to_bin(frac, frac_width):
    if frac > 1:
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
        res = res.replace("0_", "_")
    if frac_width == 0:
        res = res[:-1]
    if format:
        return f'{len(res)-1}\'b' + res
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
    bad_chars = ["b", "_"]
    for char in bad_chars:
        s = s.replace(char, "")
    res = 0
    for i, b in enumerate(s):
        if i == 0:
            res -= int(b)*2**(len(s)-1)
        else:
            res += int(b)*2**(len(s)-i-1)
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


if __name__ == '__main__':

    um_m = 10 ** -6

    f = array([13235.26131362, 16379.02884655, 20465.92663936,
               25181.57793875, 26753.46170521, 29897.22923814])
    g = array([
        24705.82111877, 30574.18718023, 38203.06306014,
        47005.61215233, 49939.79518306, 55808.16124453])

    for f_ in f:
        f_ *= um_m
        a_fp_1_8_23 = fraction_to_bin(f_, frac_width=24)
        #print(a_fp_1_8_23)

    #print()
    for g_ in g:
        g_ *= um_m
        a_fp_1_8_23 = fraction_to_bin(g_, frac_width=24)
        #print(a_fp_1_8_23)

    #exit()
    #print(bin_to_dec(a_fp_1_8_23))
    i, p = 12, 17
    n2 = dec_to_twoscompl(30.5, int_width=i, frac_width=p)
    n10 = twos_compl_to_dec("00000000011111100100", p=p)
    print(n2, n10)
    exit()
    #print(a_fp_1_8_23)

    #b_fp_1_8_23 = "0_" + int_to_bin(b, int_width=8) + "_" + fraction_to_bin(b, frac_width=23)
    #[0.01619003 0.3079267  0.11397636 0.13299026 0.05960753 0.08666484]
    #print(int_to_bin(15, 10))
    #print(fraction_to_bin(0.9, 6))
    #real = real_to_bin(0.764776164690310) #  0.7647761646903104
    #print(unsigned_dec_to_bin(1.07608, 1, 15))
    #print(real)
    #print(pi2_inv)
    #print(bin_to_dec(pi2_inv))
    #print(1/(2*np.pi))
    #bin_str = real_to_bin(2.928132598615621, int_w=7, frac_w=17)
    #print(bin_str)
    #dec_n = twos_compl_to_dec("00000100110101000110", p=17)
    #print(dec_n)

    #print(bin_to_dec("0_00100110_11000111000100101100101"))
    #print(twos_compl_to_dec("11111110100010010011001111010011", p=23))
    #print(twos_compl_to_dec("11111111110011000001111110100010"))
    #print(-4 / (pi * pi))

