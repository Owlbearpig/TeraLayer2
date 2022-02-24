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
    # TODO NOT DONE
    print("dont use this method")
    import re
    sep_loc = [m.start() for m in re.finditer("_", s)]

    s = s.replace("_", "")

    return sep_loc


def dec_to_twoscompl(r, int_width=8, frac_width=23):
    if r > 0:
        return "0_" + int_to_bin(r, int_width=int_width) + "_" + fraction_to_bin(r, frac_width=frac_width)
    else:
        res = "0_" + int_to_bin(r, int_width=int_width) + "_" + fraction_to_bin(r, frac_width=frac_width)
        res = invert_bin(res)
        # TODO add one before returning result, negative case is manual ...


def real_to_bin(n):
    # TODO fix 2s complement neg vals. This is 1s complement?
    # default precision: W(32, 8, 23)
    if n < 0:
        sign_bit = "1"
    else:
        sign_bit = "0"

    return sign_bit + "_" + int_to_bin(n, int_width=8) + "_" + fraction_to_bin(n, frac_width=23)


def twos_compl_to_dec(s, p=23):
    s = s.replace("_", "")
    res = 0
    for i, b in enumerate(s):
        if i == 0:
            res -= int(b)*2**(len(s)-1)
        else:
            res += int(b)*2**(len(s)-i-1)
    return res / (2**p)


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

um_m = 10 ** -6

f = array([13235.26131362, 16379.02884655, 20465.92663936,
           25181.57793875, 26753.46170521, 29897.22923814])
g = array([
    24705.82111877, 30574.18718023, 38203.06306014,
    47005.61215233, 49939.79518306, 55808.16124453])

for f_ in f:
    f_ *= um_m
    a_fp_1_8_23 = fraction_to_bin(f_, frac_width=24)
    print(a_fp_1_8_23)

print()
for g_ in g:
    g_ *= um_m
    a_fp_1_8_23 = fraction_to_bin(g_, frac_width=24)
    print(a_fp_1_8_23)

exit()
#print(bin_to_dec(a_fp_1_8_23))
#print(a)
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
bin_str = real_to_bin(0.08666484)
print(bin_str)
dec_n = twos_compl_to_dec("000000000_10010111101111101001110", p=23)
#print(dec_n)
"00000000001001011110111110100111"
#print(bin_to_dec("0_00100110_11000111000100101100101"))
#print(twos_compl_to_dec("11111110100010010011001111010011", p=23))
#print(twos_compl_to_dec("11111111110011000001111110100010"))
#print(-4 / (pi * pi))
