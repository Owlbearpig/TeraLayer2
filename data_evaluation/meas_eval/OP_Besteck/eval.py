import itertools
from matplotlib import ticker
import numpy as np
from load_data import OPMeasurement
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, norm
from scipy.fftpack import rfft
from consts import c_thz
# from mpl_settings import *
from scipy.signal import find_peaks
from matplotlib import cm


def deconvolve(ref_td, sam_td):
    def shrinkage_factor(H):
        H = np.array(H, dtype=np.float32)
        H = np.dot(H, H.T)
        return 1 / max(rfft(H[0]))  # since H is circulant -> H.T @ H is -> eig vals with fft

    def toeplitz_j(ref_td):  # convolve(ref, y, mode="wrap") from scipy.ndimage.filters
        c = np.hstack((ref_td[:, 1], ref_td[0, 1]))
        c = toeplitz(c, c[-1::-1])
        return c[0:-1, 0:-1]

    lambda_ = 2
    eps = 1e-14
    max_iteration_count = 20  # 2000
    step_scale = 0.3
    # tau = tau_scale / norm(a, 2)

    H = toeplitz_j(ref_td)
    tau = step_scale * shrinkage_factor(H)

    def soft_threshold(v):
        ret = np.zeros(v.shape)

        id_smaller = v <= -lambda_ * tau
        id_larger = v >= lambda_ * tau
        ret[id_smaller] = v[id_smaller] + lambda_ * tau
        ret[id_larger] = v[id_larger] - lambda_ * tau

        return ret

    def calc(H, sam_td, f):
        return np.dot(H.T, np.dot(H, f) - sam_td[:, 1])

    # init variables
    m = ref_td.shape[0]
    opt_sol = [np.zeros(m), 1, 0]
    relerr = 1
    n = 0

    f = np.zeros(m)
    ssq = norm(f, 1)
    big_f = []

    while (relerr > eps) and (n < max_iteration_count):
        n += 1
        pre_calc = calc(H, sam_td, f)
        f = soft_threshold(f - tau * pre_calc)

        big_f.append(0.5 * norm(sam_td[:, 1] - H @ f, 2) ** 2 + lambda_ * norm(f, 1))
        print(f"F(f): {big_f[-1]}")

        ssq_new = norm(f, 1)
        relerr = abs(1 - ssq / ssq_new)
        if not n % 100:
            print(f"relerr: {round(relerr, int(-np.log10(relerr)) + 2)}. Iteration: {n}\n")
        ssq = ssq_new

        opt_sol[0], opt_sol[1], opt_sol[2] = f, relerr, n

    print(f"Relerr of last iteration: {relerr}. Completed iterations: {n}")
    print(f"Err. of output solution: {opt_sol[1]} at iteration: {opt_sol[2]}\n")

    return opt_sol[0]


def deconvolve_eval(point=None):
    if point is None:
        point = (7., 8.)
    measurement = OPMeasurement(area_idx=1)
    #measurement.image()
    #plt.show()

    ref_td = measurement.get_ref(normalize=False, sub_offset=True)
    x_metal = 0.0
    sam_metal_td0 = measurement.get_point(x=x_metal, y=point[1], normalize=False, sub_offset=True)
    sam_metal_td1 = measurement.get_point(x=x_metal+0.5, y=point[1], normalize=False, sub_offset=True)
    sam_metal_td2 = measurement.get_point(x=x_metal+1.0, y=point[1], normalize=False, sub_offset=True)
    sam_metal_td3 = measurement.get_point(x=x_metal+1.5, y=point[1], normalize=False, sub_offset=True)
    sam_metal_td4 = measurement.get_point(x=x_metal + 2.0, y=point[1], normalize=False, sub_offset=True)
    sam_metal_td5 = measurement.get_point(x=x_metal + 3.5, y=point[1], normalize=False, sub_offset=True)
    sam_coating_td = measurement.get_point(x=point[0], y=point[1], normalize=False, sub_offset=True)

    f_metal = deconvolve(ref_td, sam_metal_td0)
    f_coating = deconvolve(ref_td, sam_coating_td)

    dt = measurement.info["dt"]
    # t_metal, t_coating = np.argmax(f_metal) * dt, np.argmax(f_coating) * dt

    #peak_pos = f"Peak positions: (metal: {t_metal}, {t_coating}) ps"
    #d = "$d_{coating} = $" + f"{round((t_coating - t_metal) * c_thz / 2, 1)} um"
    en_plot = True
    if en_plot:
        fig, axs = plt.subplots(2, 1, constrained_layout=True)

        axs[0].plot(sam_metal_td0[:, 0], sam_metal_td0[:, 1], label=f"Metal x={x_metal}, y={point[1]}")
        #axs[0].plot(sam_metal_td1[:, 0], sam_metal_td1[:, 1], label=f"Metal x={x_metal+0.5}, y={point[1]}")
        #axs[0].plot(sam_metal_td2[:, 0], sam_metal_td2[:, 1], label=f"Metal x={x_metal+1.0}, y={point[1]}")
        axs[0].plot(sam_metal_td3[:, 0], sam_metal_td3[:, 1], label=f"Metal x={x_metal+1.5}, y={point[1]}")
        axs[0].plot(sam_metal_td4[:, 0], sam_metal_td4[:, 1], label=f"Metal x={x_metal + 2.0}, y={point[1]}")
        axs[0].plot(sam_metal_td5[:, 0], sam_metal_td4[:, 1], label=f"Metal x={x_metal + 3.5}, y={point[1]}")
        axs[0].plot(sam_coating_td[:, 0], sam_coating_td[:, 1], label=f"Coating x={point[0]}, y={point[1]}")
        axs[0].set_xlim((15, 31))
        axs[0].set_xlabel("Time (ps)")
        #axs[0].set_ylabel("Normalized amplitude")
        axs[0].set_ylabel("Amplitude (Arb. u.)")
        axs[0].legend()

        #axs[1].text(12, 0.15, peak_pos)
        #axs[1].text(12, 0.05, d)
        axs[1].plot(ref_td[:, 0], f_metal, label="Metal impulse response")
        axs[1].plot(ref_td[:, 0], f_coating, label="Coating impulse response")
        axs[1].set_xlim((0, 31))
        axs[1].set_xlabel("Delay (ps)")
        axs[1].set_ylabel("IRF")
        axs[1].legend()

        # plt.savefig(plot_file_name, dpi=300)

    peaks, _ = find_peaks(f_coating)

    return peaks + np.argmax(ref_td[:, 1])


def plane_fit(measurement):
    dx, dy = measurement.info["dx"], measurement.info["dy"]
    image = measurement.arr
    #image[image < 0] = 0
    area_bounds = [[0, 2.0], [2, 8]]
    x_idx0, x_idx1 = int(area_bounds[0][0] / dx), int(area_bounds[0][1] / dx)
    y_idx0, y_idx1 = int(area_bounds[1][0] / dy), int(area_bounds[1][1] / dy)

    area = image[x_idx0:x_idx1, y_idx0:y_idx1]

    tof_data = np.argmax((area), axis=2).transpose((1, 0))

    plt.figure("Fitted area")
    plt.imshow(tof_data)

    min_tof, max_tof = np.min(tof_data), np.max(tof_data)
    Y = (tof_data - min_tof) / (max_tof - min_tof)

    m, n = tof_data.shape  # size of the matrix

    X1, X2 = np.mgrid[:m, :n]
    #X1 = np.flip(X1)
    #X2 = np.flip(X2)

    # Regression
    X = np.hstack((np.reshape(X1, (m * n, 1)), np.reshape(X2, (m * n, 1))))
    X = np.hstack((np.ones((m * n, 1)), X))
    YY = np.reshape(Y, (m * n, 1))

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)

    print(theta)
    en_plot = True
    if en_plot:
        X1 = np.flip(X1)
        X2 = np.flip(X2)
        plt.tight_layout()
        plane = np.reshape(np.dot(X, theta), (m, n))
        print(plane.shape)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title("ToF plane fit (Metal area)")
        ax.plot_surface(X1, X2, plane)
        ax.set_xlabel("y (mm)")
        ax.set_ylabel("x (mm)")
        ax.set_zlabel("ToF (normalized)")
        ax.set_yticklabels(reversed([str(label) for label in np.arange(0.0, 3.1, 0.5)]))
        ax.set_xticklabels([str(label) for label in list(range(1, 9, 1))])
        ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)

    """
        # Subtraction
        Y_sub = Y - plane
        ax = fig.add_subplot(3, 1, 3, projection='3d')
        ax.plot_surface(X1, X2, Y_sub, rstride=1, cstride=1, cmap=jet, linewidth=0)
    """
    print(min_tof, max_tof)
    # undo scaling: (plane * (max - min) + min)
    plane_eq = lambda x, y: float(min_tof + (max_tof - min_tof) * (theta[0] + theta[2] * x + theta[1] * y))
    t = measurement.ref_td[:, 0].real
    t -= t[0]
    print(t[min_tof], t[max_tof], t[(max_tof-min_tof)], theta[0], theta[1], theta[2])

    return plane_eq


def corrected_tof_eval(measurement):
    plane_eq = plane_fit(measurement)
    """
    point = (6.0, 4.0)
    
    area_bounds = [[6, 8], [2, 8]]
    x_idx0, x_idx1 = int(area_bounds[0][0] / dx), int(area_bounds[0][1] / dx)
    y_idx0, y_idx1 = int(area_bounds[1][0] / dy), int(area_bounds[1][1] / dy)

    area = image[x_idx0:x_idx1, y_idx0:y_idx1]
    t0_idx, t1_idx = int(15 / dt), int(20 / dt)

    x_range, y_range = np.arange(6, 8, dx), np.arange(2, 8, dy)
    d = np.zeros((len(x_range), len(y_range)))
    x_min, y_min = x_range[0], y_range[0]
    points = list(itertools.product(x_range, y_range))
    for point in points:
        dx, dy, dt = measurement.info["dx"], measurement.info["dy"], measurement.info["dt"]
        #interfaces = deconvolve_eval(point)

        x_idx, y_idx = int(point[0] / dx), int(abs(point[1] - 10) / dy)
        metal_tof_fix = plane_eq(y_idx, x_idx)

        print("TOF", point, metal_tof_fix, interfaces[0])
        #d = (metal_tof_fix - interfaces[0]) * dt * c_thz
        metal_tof_fix - tof_data[x_idx, y_idx]
        d[int((point[0] - x_min) / dx), int((point[1] - y_min) / dy)] = d
        n = 0.5 * (interfaces[1] - interfaces[0]) * dt * c_thz / d
        print(f"Deconvolve peaks: {interfaces*dt} ps")
        print(f"Metal ToF: {metal_tof_fix*dt} ps, Refractive index: {n}, thickness: {d} um")
    """
    image = measurement.arr
    # image[image < 0] = 0
    image = image.transpose((1, 0, 2))
    dx, dy, dt = measurement.info["dx"], measurement.info["dy"], measurement.info["dt"]
    t0_p1_idx, t1_p1_idx = int(15 / dt), int(20 / dt)
    tof_p1_data = np.array(t0_p1_idx + np.argmax((image[:, :, t0_p1_idx:t1_p1_idx]), axis=2), dtype=float)
    t0_p2_idx, t1_p2_idx = int(20 / dt), int(30 / dt)
    tof_p2_data = t0_p2_idx + np.argmax((image[:, :, t0_p2_idx:t1_p2_idx]), axis=2)

    # deconvolve_eval(point=None)
    metal_tof = np.zeros_like(tof_p1_data, dtype=float)
    x_range, y_range = np.arange(0, metal_tof.shape[0]*dx, dx), np.arange(0, metal_tof.shape[1]*dx, dy)
    for x_idx in range(len(x_range)):
        for y_idx in range(len(y_range)):
            #val = plane_eq(x_idx, abs(y_idx - 8 // dy))
            val = plane_eq(x_idx, 8//dy-y_idx)
            metal_tof[x_idx, y_idx] = val

    metal_tof = np.flip(metal_tof, axis=1)
    img_extent = [5, 8, 3, 8]

    metal_tof = metal_tof.transpose((1, 0))
    tof_p1_data = tof_p1_data.transpose((1, 0))
    tof_p2_data = tof_p2_data.transpose((1, 0))

    d = (0.5*(metal_tof - tof_p1_data) * dt * c_thz)
    # d = tof_p1_data
    n = 0.5 * (tof_p2_data - tof_p1_data) * dt * c_thz / d
    #n = (tof_p2_data - tof_p1_data) * dt

    d = d[int(img_extent[0]/dx):int(img_extent[1]/dx), int(img_extent[2]/dy):int(img_extent[3]/dy)]
    n = n[int(img_extent[0]/dx):int(img_extent[1]/dx), int(img_extent[2]/dy):int(img_extent[3]/dy)]

    #n = 0.5 * (tof_p2_data - interfaces[0]) * dt * c_thz / d

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Layer thickness from ToF difference")
    fig.subplots_adjust(left=0.2)
    img = ax.imshow(d.transpose((1, 0)),
                    vmin=np.min(d), vmax=np.max(d),
                    origin="upper",
                    cmap=plt.get_cmap('jet'),
                    extent=img_extent)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    def fmt(x, val):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    #cbar = fig.colorbar(img, format=ticker.FuncFormatter(fmt))
    cbar = fig.colorbar(img)
    cbar.set_label(f"Thickness (um)", rotation=270, labelpad=30)

    print(np.mean(d))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.2)
    img = ax.imshow(n.transpose((1, 0)),
                    vmin=np.min(n), vmax=np.max(n),
                    origin="upper",
                    cmap=plt.get_cmap('jet'),
                    extent=img_extent)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    def fmt(x, val):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    # cbar = fig.colorbar(img, format=ticker.FuncFormatter(fmt))
    cbar = fig.colorbar(img)
    cbar.set_label(f"Refractive index", rotation=270, labelpad=30)

    return d, n


if __name__ == '__main__':
    deconvolve_eval()
    measurement = OPMeasurement(area_idx=1)
    measurement.image(type_="tof")
    measurement.image(type_="p2p")
    corrected_tof_eval(measurement)

    plt.show()
    pass
