from base_converters import bin_to_dec, unsigned_dec_to_bin, fraction_to_bin
import itertools
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.5, 2, 10**5)

plt.plot(x, 1/x)
#plt.plot(np.linspace(0, 2, len(recip_bins)), recip_bins)
plt.show()

# 1, 15
bit_strs = [''.join(i) for i in itertools.product('01', repeat=15)]
all_bit_strs = []
for bit_s in bit_strs:
    all_bit_strs.append('0_' + bit_s)
    all_bit_strs.append('1_' + bit_s)

all_bit_strs = list(sorted(all_bit_strs))
all_bit_strs.remove('0_'+15*'0')

recip_bins = []
for s in all_bit_strs:
    x = bin_to_dec(s, signed=False)
    if (x < 0.5) or (x == 0.5):
        continue

    recip_x = 1/x
    print(recip_x)
    recip_bins.append(str(recip_x)[0] + fraction_to_bin(recip_x, frac_width=15))
print(len(recip_bins))
with open('recip_lut.mem', 'w') as file:
    for b in recip_bins:
        file.write(b+'\n')

with open('recip_lut_extended.mem', 'w') as file:
    for x in np.linspace(0.000001, 1.99999, 2**16):
        if 1/x >= 2:
            x = 0.50001
        file.write(str(int(1/x)) + fraction_to_bin(1/x, frac_width=15) + '\n')


