from functions import get_phase_measured
from consts import new_mask
from explicitEvalOptimizedClean import ExplicitEval
from visualizing.simplecolormap import map_plot


if __name__ == '__main__':
    sample_idx = 10

    new_eval = ExplicitEval(data_mask=new_mask, sample_file_idx=sample_idx)

    map_plot(new_eval.combined_error)
