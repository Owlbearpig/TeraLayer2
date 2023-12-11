import matplotlib.pyplot as plt

from triple_layer import *
from matplotlib.widgets import Button, Slider


def phasors(d1_, d2_, freq_idx_=0):
    phi0 = d1_ * kz_list[freq_idx_, 1]
    x_ = np.exp(1j * 2 * phi0)
    phi1 = d2_ * kz_list[freq_idx_, 2]
    y_ = np.exp(1j * 2 * phi1)

    p0_, p1_, p2_, p3_ = c0[freq_idx_], c1[freq_idx_] * x_, c2[freq_idx_] * y_, c4[freq_idx_] * x_ * y_
    p4_, p5_, p6_, p7_ = c3[freq_idx_], c5[freq_idx_] * x_, c6[freq_idx_] * y_, c7[freq_idx_] * x_ * y_

    return np.array([p0_, p1_, p2_, p3_, p4_, p5_, p6_, p7_], dtype=complex)


def update(val):
    d1_slider_val, d2_slider_val = d1_slider.val, d2_slider.val
    phasors_ = phasors(d1_slider_val, d2_slider_val)
    for i, line in enumerate(lines):
        phi, R = np.angle(phasors_[i]), np.abs(phasors_[i])
        if i in [0, 4]:
            xdata, ydata = [0, phi], [0, R]
        else:
            R_prev, phi_prev = np.abs(phasors_[i - 1]), np.angle(phasors_[i - 1])
            xdata, ydata = [phi_prev, phi], [R_prev, R]

        line.set_xdata(xdata)
        line.set_ydata(ydata)

    fig.canvas.draw_idle()


# radar green, solid grid lines
plt.rc('grid', color='#316931', linewidth=1, linestyle='-')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

# force square figure and square axes looks better for polar, IMO
width, height = mpl.rcParams['figure.figsize']
size = min(width, height)
# make a square figure
fig = plt.figure(figsize=(size, size))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
ax.set_ylim((0, 0.35))
plt.grid(True)

ax.set_title("And there was much rejoicing!", fontsize=20)

d1_init, d2_init = d_truth[1:3]
print(d1_init, d2_init)
axd1 = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
d1_slider = Slider(
    ax=axd1,
    label="d1",
    valmin=d1[0],
    valmax=d1[-1],
    valinit=d1_init,
    orientation="vertical"
)

axd2 = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
d2_slider = Slider(
    ax=axd2,
    label="d2",
    valmin=d2[0],
    valmax=d2[-1],
    valinit=d2_init,
    orientation="vertical"
)

phasors0 = phasors(d1_init, d2_init)
bar_colors = 4 * ['blue'] + 4 * ['red']
lines = []
print(phasors0)
for i, p_ in enumerate(phasors0):
    R, phi = np.abs(p_), np.angle(p_)
    if i in [0, 4]:
        line, = ax.plot([0, phi], [0, R], c=bar_colors[i])
    else:
        R_prev, phi_prev = np.abs(phasors0[i - 1]), np.angle(phasors0[i - 1])
        line, = ax.plot([phi_prev, phi], [R_prev, R], c=bar_colors[i])

    lines.append(line)

# register the update function with each slider
d1_slider.on_changed(update)
d2_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    d1_slider.reset()
    d2_slider.reset()


button.on_clicked(reset)

# ax.vlines(np.angle(phasors_), 0, np.abs(phasors_), colors=bar_colors, zorder=3, lw=2)

"""
def plt_arrow(*args, **kwargs):
    edgecolor = "black"
    if "c" in kwargs.keys():
        edgecolor = kwargs["c"]

    plt.arrow(*args, alpha=0.5, width=0.015, facecolor='green', lw=2, zorder=5, edgecolor=edgecolor)

for i, p_ in enumerate(phasors_):
    R = np.abs(p_)
    phi = np.angle(p_)
    if i < 3:
        plt_arrow(phi, 0.0, 0.0, R, c="blue")
    else:
        plt_arrow(phi, 0.0, 0.0, R, c="red")

    # plt_arrow(0, phi, 0, 1)
"""

plt.figure()
plt.plot([0, np.sum(phasors0[:4]).real], [0, np.sum(phasors0[:4]).imag], label="first 4")
plt.plot([0, np.sum(phasors0[4:]).real], [0, np.sum(phasors0[4:]).imag], label="last 4")
plt.xlabel("real")
plt.ylabel("imag")
show()
