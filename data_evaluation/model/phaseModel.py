from functions import get_phase_measured
from consts import custom_mask_420




if __name__ == '__main__':
    sample_idx = 10

    f, r, b, s = get_phase_measured(sample_file_idx=sample_idx, mask=custom_mask_420)

