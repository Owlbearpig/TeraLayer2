from model.phaseModel import PhaseEval
from visualizing.simplecolormap import map_plot
import numpy as np

mask = np.arange(250, 880, 10)

new_model = PhaseEval(mask)
grid_vals = map_plot(new_model.phase_loss_fullrange)

np.save("phase_grid_lowres", grid_vals)
