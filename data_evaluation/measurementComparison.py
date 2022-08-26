from model.initial_tests.explicitEvalOptimizedClean import ExplicitEval
from consts import *
from results import d_best
from pathlib import Path
from functions import calc_loss, calc_full_loss
from optimization.bruteforceOptimization import _minimize_bruteforce

d0 = array([50, 600, 50]) * um_to_m
lb = d0 - array([50, 50, 50]) * um_to_m
hb = d0 + array([50, 50, 50]) * um_to_m

mask = default_mask
minimizer = _minimize_bruteforce

data_file_cnt = 100
thicknesses = np.empty((data_file_cnt, 3))
for i in range(100):
    print(f'Measurement idx: {i}')
    new_eval = ExplicitEval(data_mask=mask, sample_file_idx=i)

    #fval, x, iterations, fcalls = _minimize_neldermead(new_eval.error, d0, bounds=(lb, hb))
    fval, x, iterations, fcalls = minimizer(new_eval.error, d0, bounds=(lb, hb))

    thicknesses[i] = x

    print(x * um)
    print(f'{len(mask)} freq. loss: ', calc_loss(x))
    print('loss over full range: ', calc_full_loss(x), f'(d_best: {calc_full_loss(d_best)})\n')

np.save(str(Path('measurementComparisonResults') / f'{minimizer.__name__}-default_mask'), thicknesses)
