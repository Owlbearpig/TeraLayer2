import numpy as np
from model.explicitEvalSimple import sine
from consts import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import FormatStrFormatter

f, ax = plt.subplots(1)
x = np.linspace(-1.4*pi, 1.4*pi, 100)
y = sin(x)
ax.plot(x, sin(x), label='sin(x)')
ax.plot(x, sine(x), label='sine(x) (approximation)')

ax.xaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))

plt.title('Approximation vs sin(x)')
plt.xlabel('x')
plt.legend()
plt.show()
