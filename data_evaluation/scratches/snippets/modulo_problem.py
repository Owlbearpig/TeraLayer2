import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

s0, s1, s2, s3 = (25.588171873005, -12.352910559385, 12.352910559384998, -0.8823507542349986)

x = np.linspace(-2*pi*1.05, 2*pi*1.05, 1000)
plt.plot(x, np.sin(x))


def correct_mod(s):
    return s % (2*pi) - pi


def test_mod(s):
    if (-pi < s) and (s < pi):
        return s

    if s > 0:
        while s > 2*pi:
            s -= 2*pi
    else:
        while s < -2*pi:
            s += 2*pi
    if s < -pi:
        return s + 2*pi
    elif s > pi:
        return s - 2*pi
    else:
        return s

s_lst = [s0, s1, s2, s3]
for s in s_lst:
    print(s)
    print(correct_mod(s), np.sin(correct_mod(s)))
    print(test_mod(s), np.sin(test_mod(s)), "\n")

plt.scatter([correct_mod(s) for s in s_lst], [np.sin(correct_mod(s)) for s in s_lst], label='correct mod')
plt.scatter([test_mod(s) for s in s_lst], [np.sin(test_mod(s)) for s in s_lst], label='test mod')
plt.legend()
plt.show()


