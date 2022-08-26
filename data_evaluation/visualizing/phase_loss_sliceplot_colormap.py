from model.initial_tests.phaseModel import PhaseEval
from visualizing.simplecolormap import map_plot
import numpy as np

"""
number at end of filename is highest freq in range
loss1 = new_model.wrappedphase_loss method
else its new_model.phase_loss
"""

mask = np.arange(250, 650, 1)

new_model = PhaseEval(mask)
data_path = "" #"""phase_grid_840.npy"

if data_path:
    data = np.load(data_path)
else:
    data = None

#grid_vals = map_plot(new_model.phase_loss, data)
grid_vals = map_plot(new_model.wrappedphase_loss, data)

np.save("phase_grid_loss1", grid_vals)
