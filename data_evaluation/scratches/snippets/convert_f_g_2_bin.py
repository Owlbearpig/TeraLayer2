from numpy import array
from base_converters import dec_to_twoscompl

f = array([13235.26131362, 16379.02884655, 20465.92663936,
           25181.57793875, 26753.46170521, 29897.22923814])

g = array([
    24705.82111877, 30574.18718023, 38203.06306014,
    47005.61215233, 49939.79518306, 55808.16124453])

f = f * 10**-6
g = g * 10**-6

for n in f:
    print(dec_to_twoscompl(n))
print()
for n in g:
    print(dec_to_twoscompl(n))




