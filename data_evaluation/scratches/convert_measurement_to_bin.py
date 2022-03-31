from consts import *
from functions import format_data
from snippets.base_converters import dec_to_twoscompl

mask = custom_mask_420
sample_idx = 0

lam, R0 = format_data(mask=mask, sample_file_idx=sample_idx)
print("R0_10", R0)

dp, p = 3, 17
for R0_i in R0:
    print(f"20'b{dec_to_twoscompl(R0_i, dp, p)}, // {R0_i}")
