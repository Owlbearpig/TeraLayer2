from scratches.snippets.base_converters import twos_compl_to_dec, dec_to_twoscompl
from functools import total_ordering
import numpy as np
import time
from numba import jit
import numba as nb
from numba import types

def zero(pd, p):
    # 0
    return Bin2sComp(0, pd=pd, p=p)

def zero_as(val):
    # 0
    if isinstance(val, str):
        return "0" * len(val)
    else:
        return zero(val.pd, val.p)

def one(pd, p):
    # this is 1
    return Bin2sComp(1, pd=pd, p=p)

def one_as(val):
    # this is also 1
    if isinstance(val, str):
        return "0" * (len(val) - 1) + "1"
    else:
        return one(pd=val.pd, p=val.p)

def invert_bin_str(bs):
    return ''.join("1" if x == "0" else "0" for x in bs)


def invert_sign(bs):
    "negate and add one"
    bs_inv = invert_bin_str(bs)
    one_ = one_as(bs_inv)

    return add_bin_strs(bs_inv, one_)

@jit(types.unicode_type(types.unicode_type, types.unicode_type), nopython=True, cache=True)
def add_bin_strs(bs0, bs1):
    res = ""
    c, prev_c = 0, 0  # carry and previous carry
    for i in range(len(bs0)):
        i = len(bs0) - i - 1
        s0, s1 = ord(bs0[i]) - 48, ord(bs1[i]) - 48

        res = str(c ^ s0 ^ s1) + res
        prev_c = c
        c = int(s0 & s1 | s1 & c | c & s0)

    if c ^ prev_c:
        print("Overflow")
        #raise Exception("Overflow")

    return res


def add_2scomp(b0, b1):
    b0_bin_s, b1_bin_s = b0.bin_s, b1.bin_s

    if b0.p < b1.p:
        b0_bin_s = b0.bin_s + "0" * (b1.p - b0.p)
    else:
        b1_bin_s = b1.bin_s + "0" * (b0.p - b1.p)

    if b0.pd < b1.pd:
        b0_bin_s = b0.bin_s[0] * (b1.pd - b0.pd) + b0.bin_s
    else:
        b1_bin_s = b1.bin_s[0] * (b0.pd - b1.pd) + b1.bin_s

    return add_bin_strs(b0_bin_s, b1_bin_s)

@jit(types.unicode_type(types.unicode_type, types.unicode_type), nopython=True, cache=True)
def unsigned_mult_bs(bs0, bs1):
    acc = "0" * (len(bs0) + len(bs1))
    for i in range(1, len(bs1)):
        i = len(bs0) - i

        if (ord(bs1[i]) - 48):
            acc = "0" + add_bin_strs(acc[0:len(bs0)], bs0) + acc[len(bs0):-1]
        else:
            acc = "0" + acc[0:-1]

    return acc


def signed_mult_bs(bs0, bs1):
    sign_flag = bs0[0] + bs1[0]

    if sign_flag == "00":
        # bs0, bs1 positive, positive
        acc = unsigned_mult_bs(bs0, bs1)
        acc = "0" + acc[0:-1]
    elif sign_flag == "01":
        # bs0, bs1 positive, negative
        acc = unsigned_mult_bs(bs0, bs1)
        bs0_neg = invert_sign(bs0)
        acc = "1" + add_bin_strs(acc[0:len(bs0)], bs0_neg) + acc[len(bs0):-1]
    elif sign_flag == "10":
        # bs0, bs1 negative, positive
        acc = unsigned_mult_bs(bs1, bs0)
        bs1_neg = invert_sign(bs1)
        acc = "1" + add_bin_strs(acc[0:len(bs0)], bs1_neg) + acc[len(bs0):-1]
    else:
        # bs0, bs1 negative, negative
        bs0, bs1 = invert_sign(bs0), invert_sign(bs1)
        acc = unsigned_mult_bs(bs0, bs1)
        acc = "0" + acc[0:-1]

    return acc

def trunc(val):
    trunc_bs = val.bin_s[0:val.pd] + "0" * val.p
    return Bin2sComp(trunc_bs, pd=val.pd, p=val.p)

@total_ordering
class Bin2sComp:
    def __init__(self, val, pd=3, p=22):
        if isinstance(val, str):
            bin_s = val.replace("_", "")
        else:
            bin_s = dec_to_twoscompl(val, pd=pd, p=p)

        self.bin_s = bin_s.replace("_", "")
        self.p = p
        self.pd = pd

        self.f = twos_compl_to_dec(self.bin_s, p=p)

    def __abs__(self):
        if int(self.bin_s[0]):
            return Bin2sComp(invert_sign(self.bin_s), self.pd, self.p)
        else:
            return self

    def __eq__(self, other):
        return self.bin_s == other.bin_s

    def __len__(self):
        return self.p + self.pd

    def __neg__(self):
        return Bin2sComp(invert_sign(self.bin_s), self.pd, self.p)

    def __add__(self, other):
        if isinstance(other, Bin2sComp):
            bin3_s = add_2scomp(self, other)
        else:
            other = Bin2sComp(other, self.pd, self.p)
            bin3_s = add_2scomp(self, other)

        return Bin2sComp(bin3_s, self.pd, self.p)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Bin2sComp):
            res = add_2scomp(self, -other)
        else:
            other = Bin2sComp(other, pd=self.pd, p=self.p)
            res = add_2scomp(self, -other)

        return Bin2sComp(res, self.pd, self.p)

    def __mul__(self, other):
        if isinstance(other, Bin2sComp):
            assert (self.p == other.p) and (self.pd == other.pd)  # TODO add support (padding)
        else:
            other = Bin2sComp(other, pd=self.pd, p=self.p)

        mul_bs = signed_mult_bs(self.bin_s, other.bin_s)
        mul_bs_trunc = mul_bs[self.pd:2 * self.pd + self.p]

        return Bin2sComp(mul_bs_trunc, self.pd, self.p)

    def __rmul__(self, other):
        if isinstance(other, Bin2sComp):
            assert (self.p == other.p) and (self.pd == other.pd)  # TODO add support (padding)
        else:
            other = Bin2sComp(other, pd=self.pd, p=self.p)

        mul_bs = signed_mult_bs(self.bin_s, other.bin_s)
        mul_bs_trunc = mul_bs[self.pd:2 * self.pd + self.p]

        return Bin2sComp(mul_bs_trunc, self.pd, self.p)

    def __lt__(self, other):
        if isinstance(other, Bin2sComp):
            assert (self.p == other.p) and (self.pd == other.pd)  # TODO add support (padding)
        else:
            other = Bin2sComp(other, pd=self.pd, p=self.p)

        # https://en.wikipedia.org/wiki/Two%27s_complement#Comparison_(ordering)
        # https://stackoverflow.com/questions/5824382/enabling-comparison-for-classes
        if (self.bin_s[0] == "0") and (other.bin_s[0] == "1"):
            return False
        elif (self.bin_s[0] == "1") and (other.bin_s[0] == "0"):
            return True

        for i in range(1, len(self.bin_s)):
            if (self.bin_s[i] == "0") and (other.bin_s[i] == "1"):
                return True
            elif (self.bin_s[i] == "1") and (other.bin_s[i] == "0"):
                return False

        return False

    def __repr__(self):
        bs = self.bin_s[0:self.pd] + "_" + self.bin_s[self.pd:]
        return f"{str(len(self))}'b{bs} (base 2)" + f" // {self.f} (base 10)"


if __name__ == '__main__':
    pd, p = 8, 8
    bin0_s = dec_to_twoscompl(3.200000, pd=pd, p=p)
    bin1_s = dec_to_twoscompl(3.123456, pd=pd, p=p)
    print(bin0_s)
    print(bin1_s)

    b0 = Bin2sComp(bin0_s, pd=pd, p=p)
    b1 = Bin2sComp(bin1_s, pd=pd, p=p)
    #print(sum([b0, b1]))
    #print(0.5*b0 - 1)
    #print(False * 2 * b0)
    b3 = b0 * b1
    print(b0)
    print(b1)
    print(b3)
    exit()
    rs = np.random.random(1000)
    numbers = []
    for r in rs:
        bin_s = dec_to_twoscompl(r, pd=pd, p=p)
        numbers.append(Bin2sComp(bin_s, pd=pd, p=p))

    start = time.process_time()
    for i in range(len(rs) - 1):
        s = numbers[i] + numbers[i + 1]
    print(time.process_time() - start)

    start = time.process_time()
    for i in range(len(rs) - 1):
        s = rs[i] * rs[i + 1]
    print(time.process_time() - start)
