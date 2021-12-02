import numpy as np


def _minimize_neldermead(func, x0, maxiter=100, xatol=1e-4, fatol=1e-4, adaptive=False, bounds=None):
    fcalls = 0

    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2/dim
        psi = 0.75 - 1/(2*dim)
        sigma = 1 - 1/dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    nonzdelt = 0.05
    zdelt = 0.00025

    lower_bound, upper_bound = bounds

    N = len(x0)

    sim = np.empty((N + 1, N), dtype=x0.dtype)
    sim[0] = x0
    for k in range(N):
        y = np.array(x0, copy=True)
        if y[k] != 0:
            y[k] = (1 + nonzdelt)*y[k]
        else:
            y[k] = zdelt
        sim[k + 1] = y

    sim = np.clip(sim, lower_bound, upper_bound)

    one2np1 = list(range(1, N + 1))
    fsim = np.empty((N + 1,), float)

    for k in range(N + 1):
        fcalls += 1
        fsim[k] = func(sim[k])

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)

    iterations = 1
    while (iterations < maxiter):
        if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
            break

        xbar = np.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        xr = np.clip(xr, lower_bound, upper_bound)

        fxr = func(xr)
        fcalls += 1
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            xe = np.clip(xe, lower_bound, upper_bound)

            fxe = func(xe)
            fcalls += 1

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    xc = np.clip(xc, lower_bound, upper_bound)

                    fxc = func(xc)
                    fcalls += 1

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    xcc = np.clip(xcc, lower_bound, upper_bound)

                    fxcc = func(xcc)
                    fcalls += 1

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        sim[j] = np.clip(sim[j], lower_bound, upper_bound)
                        fsim[j] = func(sim[j])
                        fcalls += 1

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
        iterations += 1

    x = sim[0]
    fval = np.min(fsim)

    return fval, x, iterations, fcalls

