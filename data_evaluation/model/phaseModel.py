from functions import get_phase_measured
from consts import new_mask, all_freqs_lowend, array, um_to_m, np, GHz, full_range_mask, pi
from model.explicitEvalOptimizedClean import ExplicitEval
from visualizing.simplecolormap import map_plot
import matplotlib.pyplot as plt


class PhaseEval():

    def __init__(self, mask=None):
        sample_idx = 10
        if mask is None:
            mask = full_range_mask  # new_mask
        self.new_model = ExplicitEval(data_mask=mask, sample_file_idx=sample_idx)

        f, r, b, s = get_phase_measured(sample_file_idx=sample_idx, mask=mask)
        self.phase_sam_wrapped = s.copy()

        r, s = np.unwrap(r), np.unwrap(s)

        fit_slice = (f > 475 * GHz) * (f < 575 * GHz)  # nice(linear, no jumps) part of measurement

        pr, ps = np.polyfit(f[fit_slice] / GHz, r[fit_slice], 1), np.polyfit(f[fit_slice] / GHz, s[fit_slice], 1)
        r -= pr[1]
        s -= ps[1]


        self.phase_diff = s - r

    def phase_loss_fullrange(self, p):
        r = self.new_model.explicit_reflectance(p, return_magn=False, return_r=True)
        phase = np.unwrap(np.angle(-1 * r))

        offset = 0
        unwrapped_phase = phase.copy()
        for i in range(1, len(phase)):
            if np.abs(phase[i - 1] - phase[i]) > np.pi * 0.9:
                offset += np.pi
            unwrapped_phase[i] -= offset

        p = np.polyfit(self.new_model.freqs / GHz, unwrapped_phase, 1)

        unwrapped_phase -= p[1]

        return np.sum((self.phase_diff - unwrapped_phase) ** 2)

    def phase_loss(self, p):
        r = self.new_model.explicit_reflectance(p, return_magn=False, return_r=True)
        phase = np.angle(r)

        shift = np.abs(self.phase_diff[0] - np.unwrap(phase)[0])
        phase = np.unwrap(phase) - shift

        return np.sum((self.phase_diff - phase) ** 2)

    def wrappedphase_loss(self, p):
        r = self.new_model.explicit_reflectance(p, return_magn=False, return_r=True)
        phase = np.angle(r)

        return np.sum((self.phase_sam_wrapped - phase) ** 2)


if __name__ == '__main__':
    dotsize = 2
    sample_idx = 10
    mask = full_range_mask  # new_mask
    mask = np.arange(0, 1500, 1)
    # mask = np.arange(475, 575, 1)
    # fit slice measured: 475 - 575 GHz

    new_model = ExplicitEval(data_mask=mask, sample_file_idx=sample_idx)

    # p = array([44, 630, 44]) * um_to_m
    # p = array([44, 74, 44]) * um_to_m
    p = array([106.42211055276383, 743.9748743718594, 86.34170854271358]) * um_to_m
    p = array([40, 640, 75]) * um_to_m
    r = new_model.explicit_reflectance(p, return_magn=False, return_r=True)
    print(new_model.freqs)
    plt.plot(new_model.freqs, np.unwrap(np.angle(r)))
    plt.show()


    phase = np.angle(r)  # np.unwrap(np.angle(-1*r))

    offset = 0
    unwrapped_phase = phase.copy()
    for i in range(1, len(phase)):
        if np.abs(phase[i - 1] - phase[i]) > np.pi * 0.9:
            offset += np.pi
        unwrapped_phase[i] -= offset

    p = np.polyfit(new_model.freqs / GHz, unwrapped_phase, 1)

    unwrapped_phase -= p[1]
    phase_fit = p[0] * new_model.freqs / GHz

    f, r, b, s = get_phase_measured(sample_file_idx=sample_idx, mask=mask)

    def single_spike_filter(phase_in):
        phase_out = phase_in.copy()
        for i in range(1, len(phase_diff) - 1):
            if (np.abs(phase_diff[i - 1] - phase_diff[i]) > pi) * (np.abs(phase_diff[i] - phase_diff[i + 1]) > pi):
                if phase_diff[i] > phase_diff[i - 1]:
                    phase_out[i] -= 2 * pi
                else:
                    phase_out[i] += 2 * pi
        return phase_out

    def phase_filter(phase_in):
        phase_out = phase_in.copy()
        for i in range(1, len(phase_out)-1):
            if np.abs(phase_out[i-1] - phase_out[i]) > pi:
                if (phase_out[i-1] - phase_out[i]) < 0:  # jump up
                    phase_out[i] -= 2*pi
                else:  # jump down
                    phase_out[i] += 2*pi

        return phase_out

    phase_diff = s - r
    phase_diff_filtered = np.unwrap(phase_diff)
    #phase_spike_filtered = single_spike_filter(phase_diff)
    """
    plt.figure()
    plt.title("filter")
    plt.plot(f / GHz, phase_diff, label="s - r, pure")
    plt.plot(f / GHz, phase_diff_filtered, label="s - r, filtered")
    plt.plot(f / GHz, np.unwrap(phase_diff), label="s - r, unwrap")
    plt.legend()
    plt.show()
    exit()
    """

    # r, s = np.unwrap(r), np.unwrap(s)

    fit_slice = (f > 475 * GHz) * (f < 575 * GHz)

    pr, ps = np.polyfit(f[fit_slice] / GHz, r[fit_slice], 1), np.polyfit(f[fit_slice] / GHz, s[fit_slice], 1)
    # r -= pr[1]
    # s -= ps[1]

    """
    plt.figure()
    plt.scatter(f / GHz, r, label=f'ref measured shifted(b={round(pr[1], 3)})', s=dotsize, color="grey")
    plt.scatter(f / GHz, s, label=f'sample Kopf_1x_{sample_idx + 1:04} shifted(b={round(pr[1], 3)})', s=dotsize,
                color="black")
    # plt.plot(f / GHz, pr[0] * f / GHz, label='lin. interpol ref')
    # plt.plot(f / GHz, ps[0] * f / GHz, label='lin. interpol sam')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('phase (rad)')
    plt.legend()
    """

    plt.figure()
    shift = np.abs(phase_diff_filtered[0]-np.unwrap(phase)[0])
    plt.scatter(new_model.freqs / GHz, np.unwrap(phase) - shift, label="np.angle(r_model)", s=dotsize, color="red")
    plt.plot(f / GHz, phase_diff_filtered, label='measured, sam - ref filtered')
    new_model = PhaseEval(mask)
    #plt.plot(f / GHz, new_model.phase_diff, label='lin interpol shifted phase')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('phase (rad)')
    plt.legend()
    plt.show()
