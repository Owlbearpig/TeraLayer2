import numpy as np
from scipy.special import huber
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 500)
deltas = [1, 2, 3]
linestyles = ["dashed", "dotted", "dashdot"]
fig, ax = plt.subplots()
combined_plot_parameters = list(zip(deltas, linestyles))
for delta_, style in combined_plot_parameters:
    ax.plot(x, huber(delta_, x), label=f"$\delta={delta_}$", ls=style)
ax.legend(loc="upper center")
ax.set_xlabel("$x$")
ax.set_title("Huber loss function $h_{\delta}(x)$")
ax.set_xlim(-4, 4)
ax.set_ylim(0, 8)
plt.show()

