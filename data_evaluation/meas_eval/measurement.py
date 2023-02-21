from tds.main import load_data as load_data_tds
from cw.main import load_data as load_data_cw
from functions import filtering, do_fft, do_ifft, window


class Measurement:
    def __init__(self, sam_idx=None):
        self.sam_idx = sam_idx
        self.ref_td, self.sam_td = None, None
        self.ref_fd, self.sam_fd = None, None

    def load_measurement(self, tds_data=True, en_window=False):
        if tds_data:
            self.ref_td, self.sam_td = load_data_tds(sam_idx=self.sam_idx, signal_shift=-5)

            en_filter = False
            if en_filter:
                self.ref_td = filtering(self.ref_td, filt_type="bp", wn=(1.0, 2.3))
                self.sam_td = filtering(self.sam_td, filt_type="bp", wn=(1.0, 2.3))

            if en_window:
                self.ref_td = window(self.ref_td, win_len=None, win_start=None, first_pulse=True)
                self.sam_td = window(self.sam_td, win_len=None, win_start=None, first_pulse=True)

            self.ref_fd, self.sam_fd = do_fft(self.ref_td), do_fft(self.sam_td)
        else:
            self.ref_fd, self.sam_fd = load_data_cw(sam_idx=self.sam_idx)

            self.ref_td, self.sam_td = do_ifft(self.ref_fd), do_ifft(self.sam_fd)

        return self.ref_td, self.sam_td, self.ref_fd, self.sam_fd


if __name__ == '__main__':
    pass
