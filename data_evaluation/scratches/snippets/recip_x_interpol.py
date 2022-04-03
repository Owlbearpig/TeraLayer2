from base_converters import twos_compl_to_dec
import matplotlib.pyplot as plt
import numpy as np

mem_file = "recip_lut_extended.mem"

with open(mem_file) as file:
    p = 14
    numbers = []
    lines = file.readlines()
    for line in lines:
        numbers.append(abs(twos_compl_to_dec(line, p)))
    lut = np.array(numbers)
    x = np.linspace(0.5, 2.5, len(lut))

    a = lut*x - 1

    #taylor = 1 - a + a**2# - a**3 + a**4 - a**5 + a**6 # working
    R = lut
    taylor = (3/2 - R*x)**2 + 3/4
    test = lut * taylor
    breakpoint()
    print("mean difference (lut - 1/x): ", np.mean(lut - 1 / x))
    print("max. difference abs(lut - 1/x): ", np.max(abs(lut - 1 / x)), 2 ** (-14))

    print("mean. difference abs(taylor - 1/x): ", np.mean(test - 1 / x))
    print("max. difference abs(taylor - 1/x): ", np.max(abs(test - 1 / x)))

    """
    plt.plot(x, numbers+2**(-15), label="test")
    approx = numbers + 2 ** (-15)
    approx_ma = np.array([(approx[i]+approx[i+1])/2 for i in range(1, len(approx)-1)])

    print("mean difference (approx_ma - 1/x): ", np.mean(approx_ma - 1 / x[1:2**16-1]))
    print("max. difference abs(approx_ma - 1/x): ", np.max(abs(approx_ma - 1 / x[1:2**16-1])), 2 ** (-14))

    plt.plot(x[1:len(x)-1], approx_ma, label="test approx_ma")
    """

    plt.plot(x, 1 / x, label='1/x np')
    plt.plot(x, lut, label='lut')
    plt.plot(x, test, label='taylor test')
    plt.plot()
    plt.xlabel("x")
    plt.ylabel("1/x")
    plt.legend()
    plt.show()

    #plt.plot(x, abs(lut - 1 / x), label='difference lut-1/x')
    #plt.xlabel("x")
    #plt.legend()
    #plt.show()
