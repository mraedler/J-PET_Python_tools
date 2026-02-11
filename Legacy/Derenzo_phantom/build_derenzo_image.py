"""
Phantom

Author: Martin Rädler
"""
# Libraries
import sys
from pickle import dump
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# import os
# os.environ['QT_XCB_GL_INTEGRATION'] = 'none'

# Auxiliary functions
from CASToR.read_interfile import read_interfile
from CASToR.build_derenzo_phantom import get_derenzo_parameters, get_triangles_parameters
from CASToR.vis import vis_3d


def main():
    # Coordinate system from the reconstruction
    x, y, _, _ = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/derenzo/subsets/sub_1/'
                                'img_TB_brain_tot_all_GATE_it1.hdr', return_grid=True)

    x_grid = np.append(x - (x[1] - x[0]) / 2, 3 * x[-1] / 2 - x[-2] / 2)
    y_grid = np.append(y - (y[1] - y[0]) / 2, 3 * y[-1] / 2 - y[-2] / 2)

    # Get the Derenzo parameters
    x_peaks, y_peaks, r_peaks, x_valleys, y_valleys, r_valleys = get_derenzo_parameters(scaling_factor=2., return_valleys=True)

    dump(x_peaks, open(sys.path[0] + '/Derenzo_pixelated/x_peaks.pkl', 'wb'))
    dump(y_peaks, open(sys.path[0] + '/Derenzo_pixelated/y_peaks.pkl', 'wb'))
    dump(r_peaks, open(sys.path[0] + '/Derenzo_pixelated/r_peaks.pkl', 'wb'))

    # dump(x_valleys, open(sys.path[0] + '/Derenzo_pixelated/x_valleys.pkl', 'wb'))
    # dump(y_valleys, open(sys.path[0] + '/Derenzo_pixelated/y_valleys.pkl', 'wb'))
    # dump(r_valleys, open(sys.path[0] + '/Derenzo_pixelated/r_valleys.pkl', 'wb'))

    dump(x_valleys, open(sys.path[0] + '/Derenzo_pixelated/x_valleys_parzych.pkl', 'wb'))
    dump(y_valleys, open(sys.path[0] + '/Derenzo_pixelated/y_valleys_parzych.pkl', 'wb'))
    dump(r_valleys, open(sys.path[0] + '/Derenzo_pixelated/r_valleys_parzych.pkl', 'wb'))

    # Construct the images
    img_peaks = np.zeros((x.size, y.size, len(x_peaks)))
    img_valleys = np.zeros((x.size, y.size, len(x_peaks)))
    for ii in range(0, len(x_peaks)):
        for jj in range(0, x_peaks[ii].size):
            img_peaks[:, :, ii] += pixelated_ellipse(x_grid, y_grid, x_peaks[ii][jj], y_peaks[ii][jj], r_peaks[ii][jj], r_peaks[ii][jj], 0., vis=True)

        for jj in range(0, x_valleys[ii].size):
            img_valleys[:, :, ii] += pixelated_ellipse(x_grid, y_grid, x_valleys[ii][jj], y_valleys[ii][jj], r_valleys[ii][jj], r_valleys[ii][jj], 0., vis=True)

    # vis_3d(img_peaks)
    # vis_3d(img_valleys)
    vis_3d(img_peaks - img_valleys)

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.imshow(np.sum(img_peaks, axis=-1).T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_ylabel(r'$y$ [mm]')
    ax1.imshow(np.sum(img_valleys, axis=-1).T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    ax1.set_xlabel(r'$x$ [mm]')
    # ax1.set_ylabel(r'$y$ [mm]')
    plt.show()

    np.save(sys.path[0] + '/Derenzo_pixelated/x.npy', x)
    np.save(sys.path[0] + '/Derenzo_pixelated/y.npy', y)
    np.save(sys.path[0] + '/Derenzo_pixelated/x_grid.npy', x_grid)
    np.save(sys.path[0] + '/Derenzo_pixelated/y_grid.npy', y_grid)
    np.save(sys.path[0] + '/Derenzo_pixelated/img_peaks.npy', img_peaks)
    # np.save(sys.path[0] + '/Derenzo_pixelated/img_valleys.npy', img_valleys)
    np.save(sys.path[0] + '/Derenzo_pixelated/img_valleys_parzych.npy', img_valleys)

    return 0


def main2():
    # Coordinate system from the reconstruction
    x, y, _, _ = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/derenzo/subsets/sub_1/'
                                'img_TB_brain_tot_all_GATE_it1.hdr', return_grid=True)

    x_grid = np.append(x - (x[1] - x[0]) / 2, 3 * x[-1] / 2 - x[-2] / 2)
    y_grid = np.append(y - (y[1] - y[0]) / 2, 3 * y[-1] / 2 - y[-2] / 2)

    # Get the Derenzo parameters
    x_peaks, y_peaks, r_peaks, x_valleys, y_valleys, r_valleys = get_triangles_parameters(return_valleys=True)

    # Construct the images
    img_peaks = np.zeros((x.size, y.size))
    img_valleys = np.zeros((x.size, y.size))
    for ii in range(0, len(x_peaks)):
        for jj in range(0, x_peaks[ii].size):
            img_peaks += pixelated_ellipse(x_grid, y_grid, x_peaks[ii][jj], y_peaks[ii][jj], r_peaks[ii][jj], r_peaks[ii][jj], 0., vis=True)

        for jj in range(0, x_valleys[ii].size):
            img_valleys += pixelated_ellipse(x_grid, y_grid, x_valleys[ii][jj], y_valleys[ii][jj], r_valleys[ii][jj], r_valleys[ii][jj], 0., vis=True)

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.imshow(img_peaks.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_ylabel(r'$y$ [mm]')
    ax1.imshow(img_valleys.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    ax1.set_xlabel(r'$x$ [mm]')
    # ax1.set_ylabel(r'$y$ [mm]')
    plt.show()

    np.save(sys.path[0] + '/Triangles_pixelated/x.npy', x)
    np.save(sys.path[0] + '/Triangles_pixelated/y.npy', y)
    np.save(sys.path[0] + '/Triangles_pixelated/x_grid.npy', x_grid)
    np.save(sys.path[0] + '/Triangles_pixelated/y_grid.npy', y_grid)

    np.save(sys.path[0] + '/Triangles_pixelated/img_peaks_3.2mm_31.npy', img_peaks)
    np.save(sys.path[0] + '/Triangles_pixelated/img_valleys_3.2mm_31.npy', img_valleys)

    return 0


def main3():
    x_grid = np.arange(-80, 80 + 1)
    y_grid = np.arange(-80, 80 + 1)
    # x_grid = np.arange(-96, 96 + 1)
    # y_grid = np.arange(-96, 96 + 1)
    # x_grid = np.arange(-64, 64 + 1)
    # y_grid = np.arange(-64, 64 + 1)

    x = (x_grid[1:] + x_grid[:-1]) / 2
    y = (y_grid[1:] + y_grid[:-1]) / 2

    # Get the Derenzo parameters
    x_peaks, y_peaks, r_peaks, x_valleys, y_valleys, r_valleys = get_derenzo_parameters(scaling_factor=3., return_valleys=True, visualize=False)
    radii = np.array([rr[0] for rr in r_peaks])

    np.save(sys.path[0] + '/Derenzo_pixelated_3/x_peaks.npy', np.array(x_peaks, dtype=object), allow_pickle=True)
    np.save(sys.path[0] + '/Derenzo_pixelated_3/y_peaks.npy', np.array(y_peaks, dtype=object), allow_pickle=True)
    np.save(sys.path[0] + '/Derenzo_pixelated_3/x_valleys.npy', np.array(x_valleys, dtype=object), allow_pickle=True)
    np.save(sys.path[0] + '/Derenzo_pixelated_3/y_valleys.npy', np.array(y_valleys, dtype=object), allow_pickle=True)

    sys.exit()

    # Construct the images
    img_peaks = np.zeros((x.size, y.size, len(x_peaks)))
    img_valleys = np.zeros((x.size, y.size, len(x_peaks)))
    for ii in range(0, len(x_peaks)):
        for jj in range(0, x_peaks[ii].size):
            img_peaks[:, :, ii] += pixelated_ellipse(x_grid, y_grid, x_peaks[ii][jj], y_peaks[ii][jj], r_peaks[ii][jj], r_peaks[ii][jj], 0., vis=True)

        for jj in range(0, x_valleys[ii].size):
            img_valleys[:, :, ii] += pixelated_ellipse(x_grid, y_grid, x_valleys[ii][jj], y_valleys[ii][jj], r_valleys[ii][jj], r_valleys[ii][jj], 0., vis=True)

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.imshow(np.sum(img_peaks, axis=-1).T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_ylabel(r'$y$ [mm]')
    ax1.imshow(np.sum(img_valleys, axis=-1).T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    ax1.set_xlabel(r'$x$ [mm]')
    # ax1.set_ylabel(r'$y$ [mm]')
    plt.show()

    np.save(sys.path[0] + '/Derenzo_pixelated_3/x.npy', x)
    np.save(sys.path[0] + '/Derenzo_pixelated_3/y.npy', y)
    np.save(sys.path[0] + '/Derenzo_pixelated_3/x_grid.npy', x_grid)
    np.save(sys.path[0] + '/Derenzo_pixelated_3/y_grid.npy', y_grid)
    np.save(sys.path[0] + '/Derenzo_pixelated_3/radii.npy', radii)
    np.save(sys.path[0] + '/Derenzo_pixelated_3/img_peaks.npy', img_peaks)
    # np.save(sys.path[0] + '/Derenzo_pixelated_3/img_valleys.npy', img_valleys)
    np.save(sys.path[0] + '/Derenzo_pixelated_3/img_valleys_parzych.npy', img_valleys)

    print(x_grid.size)

    return 0


def get_ellipse_coefficients_quadratic(x_c, y_c, a, b, theta):
    # Ellipse equation taken from:
    # https://en.wikipedia.org/wiki/Ellipse#General_ellipse

    f_1 = a ** 2 * np.sin(theta) ** 2 + b ** 2 * np.cos(theta) ** 2
    f_2 = 2 * (b ** 2 - a ** 2) * np.sin(theta) * np.cos(theta)
    f_3 = a ** 2 * np.cos(theta) ** 2 + b ** 2 * np.sin(theta) ** 2
    f_4 = -2 * f_1 * x_c - f_2 * y_c
    f_5 = -f_2 * x_c - 2 * f_3 * y_c
    f_6 = f_1 * x_c ** 2 + f_2 * x_c * y_c + f_3 * y_c ** 2 - a ** 2 * b ** 2

    return f_1, f_2, f_3, f_4, f_5, f_6


def t_parametrization(x_c, y_c, a, b, theta, t):
    x = a * np.cos(theta) * np.cos(t) - b * np.sin(theta) * np.sin(t) + x_c
    y = a * np.sin(theta) * np.cos(t) + b * np.cos(theta) * np.sin(t) + y_c
    return x, y


def get_x_grid_intersection_times(x_c, y_c, a, b, theta, x_grid, tol=1e-12):
    # Estimate, which x_grid values can intersect with the ellipse
    a_p = np.sqrt(a ** 2 * np.cos(theta) ** 2 + b ** 2 * np.sin(theta) ** 2)
    diff = a_p ** 2 - (x_grid - x_c) ** 2
    val_x = diff >= 0

    # Calculate the corresponding t-values
    rad = a ** 2 * np.cos(theta) ** 2 * diff
    t_1 = np.arcsin((- b * (x_grid[val_x] - x_c) * np.sin(theta) - np.sqrt(rad[val_x])) / a_p ** 2)
    t_2 = np.arcsin((- b * (x_grid[val_x] - x_c) * np.sin(theta) + np.sqrt(rad[val_x])) / a_p ** 2)
    tt = np.concatenate((t_1, t_2, np.pi - t_1, np.pi - t_2))

    # Remove invalid solutions
    x_grid_tt, _ = t_parametrization(x_c, y_c, a, b, theta, tt)
    val_t = np.abs(x_grid_tt - np.tile(x_grid[val_x], 4)) < tol
    tt = tt[val_t]

    # Map to [0, 2*pi]
    tt = np.mod(tt, 2 * np.pi)

    return tt


def get_y_grid_intersection_times(x_c, y_c, a, b, theta, y_grid, tol=1e-12):
    # Estimate, which y_grid values can intersect with the ellipse
    b_p = np.sqrt(a ** 2 * np.sin(theta) ** 2 + b ** 2 * np.cos(theta) ** 2)
    diff = b_p ** 2 - (y_grid - y_c) ** 2
    val_y = diff >= 0

    # Calculate the corresponding t-values
    rad = a ** 2 * np.sin(theta) ** 2 * diff
    t_1 = np.arcsin((b * (y_grid[val_y] - y_c) * np.cos(theta) - np.sqrt(rad[val_y])) / b_p ** 2)
    t_2 = np.arcsin((b * (y_grid[val_y] - y_c) * np.cos(theta) + np.sqrt(rad[val_y])) / b_p ** 2)
    tt = np.concatenate((t_1, t_2, np.pi - t_1, np.pi - t_2))

    # Remove invalid solutions
    _, y_grid_tt = t_parametrization(x_c, y_c, a, b, theta, tt)
    val_t = np.abs(y_grid_tt - np.tile(y_grid[val_y], 4)) < tol
    tt = tt[val_t]

    # Map to [0, 2*pi]
    tt = np.mod(tt, 2 * np.pi)

    return tt


def get_x_min_max_times(x_c, y_c, a, b, theta, dec=12):
    a_p = np.sqrt(a ** 2 * np.cos(theta) ** 2 + b ** 2 * np.sin(theta) ** 2)
    t = np.arcsin([-b * np.sin(theta) / a_p, b * np.sin(theta) / a_p])
    t = np.concatenate((t, np.pi - t))

    # Map to [0, 2 * pi]
    t = np.mod(t, 2 * np.pi)

    # Remove invalid solutions
    val_t = np.abs(-a * np.cos(theta) * np.sin(t) - b * np.sin(theta) * np.cos(t)) < 10 ** (-dec)
    t = t[val_t]

    # Remove duplicate solutions
    _, idx = np.unique(np.round(t, decimals=dec), return_index=True)
    t = t[idx]

    # Descending sorting with respect to x
    x, _ = t_parametrization(x_c, y_c, a, b, theta, t)
    idx = np.flip(np.argsort(x))
    t = t[idx]

    return t


def get_y_min_max_times(x_c, y_c, a, b, theta, dec=12):
    b_p = np.sqrt(b ** 2 * np.cos(theta) ** 2 + a ** 2 * np.sin(theta) ** 2)
    t = np.arcsin([-b * np.cos(theta) / b_p, b * np.cos(theta) / b_p])
    t = np.concatenate((t, np.pi - t))

    # Map to [0, 2*pi]
    t = np.mod(t, 2 * np.pi)

    # Remove invalid solutions
    val_t = np.abs(-a * np.sin(theta) * np.sin(t) + b * np.cos(theta) * np.cos(t)) < 10 ** (-dec)
    t = t[val_t]

    # Remove duplicate solutions
    _, idx = np.unique(np.round(t, decimals=dec), return_index=True)
    t = t[idx]

    # Descending sorting with respect to y
    _, y = t_parametrization(x_c, y_c, a, b, theta, t)
    idx = np.flip(np.argsort(y))
    t = t[idx]

    return t


def integral_quadratic_equation(f_1, f_2, f_3, f_4, f_5, f_6, x, bl):
    temp0 = f_2 ** 2 - 4 * f_1 * f_3 + 0j
    temp1 = f_5 ** 2 + 2 * f_2 * f_5 * x + f_2 ** 2 * np.square(x) - 4 * f_3 * (f_6 + x * (f_4 + f_1 * x)) + 0*x*1j
    temp2 = -2 * f_3 * (f_4 + 2 * f_1 * x) + f_2 * (f_5 + f_2 * x) + np.sqrt(temp0 * temp1)

    # x = x + 0*x*1j

    if bl:
        k = -1 / (2 * f_3) * (f_5 * x + f_2 / 2 * np.square(x) + (-2 * f_3 * (f_4 + 2 * f_1 * x) + f_2 * (f_5 + f_2 * x)) / (
                        2 * temp0) * np.sqrt(temp1) - 2 * f_3 / temp0 ** (3 / 2) * (
                                    -f_2 * f_4 * f_5 + f_1 * f_5 ** 2 + f_2 ** 2 * f_6 + f_3*(f_4 ** 2 - 4 * f_1 * f_6)) * np.log(temp2))
    else:
        k = -1 / (2 * f_3) * (f_5 * x + f_2 / 2 * np.square(x) - (-2 * f_3 * (f_4 + 2 * f_1 * x) + f_2 * (f_5 + f_2 * x)) / (
                        2 * temp0) * np.sqrt(temp1) + 2 * f_3 / temp0 ** (3 / 2) * (
                                    -f_2 * f_4 * f_5 + f_1 * f_5 ** 2 + f_2 ** 2 * f_6 + f_3*(f_4 ** 2 - 4 * f_1 * f_6)) * np.log(temp2))

    return k


def cyclic_extraction(idx_0, idx_1, arr):
    if idx_0 <= idx_1:
        return arr[idx_0:idx_1 + 1]
    else:
        return np.concatenate((arr[idx_0:], arr[:idx_1 + 1]))


def get_coordinates_and_indices(x_c, y_c, a, b, theta, x_grid, y_grid, t):
    # Edge coordinates
    xe, ye = t_parametrization(x_c, y_c, a, b, theta, t)

    # Center coordinates
    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2

    # Indices
    idx, idy = np.digitize(xc, x_grid) - 1, np.digitize(yc, y_grid) - 1
    idx_lin = np.ravel_multi_index((idx, idy), (x_grid.size - 1, y_grid.size - 1))

    return xe, ye, xc, yc, idx, idy, idx_lin


def fill_closed_domain(x_grid, y_grid, img):
    #
    mask = img != 0
    idx_left = np.argmax(mask, axis=0)

    idx_right = np.argmax(np.flip(mask, axis=0), axis=0)
    idx_right = x_grid.size - 2 - idx_right

    idy = np.arange(y_grid.size - 1)

    # Remove rows with all zeros
    all_zeros = ~np.any(mask, axis=0)
    idx_left = idx_left[~all_zeros]
    idx_right = idx_right[~all_zeros]
    idy = idy[~all_zeros]

    #
    idx_fill, idy_fill = [], []

    for ii in range(idx_left.size):
        idx_fill_candidates = np.arange(idx_left[ii], idx_right[ii] + 1)
        idy_fill_candidates = np.ones(idx_fill_candidates.size, dtype=int) * idy[ii]
        criterion = img[idx_fill_candidates, idy_fill_candidates] == 0
        idx_fill.append(idx_fill_candidates[criterion])
        idy_fill.append(idy_fill_candidates[criterion])

    idx_fill = np.concatenate(idx_fill)
    idy_fill = np.concatenate(idy_fill)

    # Pixel area
    dx_mesh, dy_mesh = np.meshgrid(np.diff(x_grid), np.diff(y_grid), indexing='ij')
    pixel_area = dx_mesh * dy_mesh

    img[idx_fill, idy_fill] = pixel_area[idx_fill, idy_fill]

    return img, pixel_area


def pixelated_ellipse(x_grid, y_grid, x_c, y_c, a, b, theta, vis=False):
    # Allocate
    res = np.zeros((x_grid.size - 1, y_grid.size - 1))

    # Change the ellipse parameters to the coefficients of the equivalent quadratic equation
    f_1, f_2, f_3, f_4, f_5, f_6 = get_ellipse_coefficients_quadratic(x_c, y_c, a, b, theta)

    #
    t_x = get_x_grid_intersection_times(x_c, y_c, a, b, theta, x_grid)
    t_y = get_y_grid_intersection_times(x_c, y_c, a, b, theta, y_grid)
    t_mx = get_x_min_max_times(x_c, y_c, a, b, theta)
    t_my = get_y_min_max_times(x_c, y_c, a, b, theta)
    t = np.concatenate((t_mx, t_my, t_x, t_y))

    _, idx, inv = np.unique(np.round(t, decimals=12), return_index=True, return_inverse=True)
    t = t[idx]

    # Indices of the extrema
    x_ex, y_ex, _, _, _, _, _ = get_coordinates_and_indices(x_c, y_c, a, b, theta, x_grid, y_grid, t)
    idx_ex, idy_ex = np.digitize(x_ex[inv[:4]], x_grid) - 1, np.digitize(y_ex[inv[:4]], y_grid) - 1
    idx_lin_ex = np.ravel_multi_index((idx_ex, idy_ex), (x_grid.size - 1, y_grid.size - 1))


    """ Quadrant I """
    t_q1 = cyclic_extraction(inv[0], inv[2], t)
    xe_q1, ye_q1, xc_q1, yc_q1, idx_q1, idy_q1, idx_lin_q1 = get_coordinates_and_indices(x_c, y_c, a, b, theta, x_grid, y_grid, t_q1)

    itg_q1 = (integral_quadratic_equation(f_1, f_2, f_3, f_4, f_5, f_6, xe_q1[1:], 0) -
              integral_quadratic_equation(f_1, f_2, f_3, f_4, f_5, f_6, xe_q1[:-1], 0)).real

    box_height_q1 = y_grid[idy_q1]
    box_height_q1[0] = ye_q1[0]
    # box_height_q1[box_height_q1 < ye_q1[0]] = ye_q1[0]

    box_q1 = box_height_q1 * np.diff(xe_q1)
    res[idx_q1, idy_q1] += box_q1 - itg_q1  # technically the other way around, but integrating from right to left

    # Add the rectangular shapes within the pixels
    left_not_on_x_grid_q1 = ~ np.isin(np.round(xe_q1[1:], decimals=12), np.round(x_grid, decimals=12), assume_unique=True)
    left_not_on_x_grid_q1[-1] = False

    # Index of the grid to the right of it
    idx_g_q1 = np.searchsorted(x_grid, xe_q1, side='right')
    x0_q1 = x_grid[idx_g_q1[1:] - 1]
    # x0_q1[x0_q1 < xe_q1[-1]] = xe_q1[-1]
    x1_q1 = xe_q1[1:]
    dx_q1 = x1_q1 - x0_q1

    y0_q1 = y_grid[idy_q1]
    y0_q1[0] = ye_q1[0]
    # y0_q1[y0_q1 < ye_q1[0]] = ye_q1[0]
    y1_q1 = y_grid[idy_q1 + 1]
    dy_q1 = y1_q1 - y0_q1

    res[idx_q1[left_not_on_x_grid_q1], idy_q1[left_not_on_x_grid_q1]] += (dx_q1 * dy_q1)[left_not_on_x_grid_q1]

    """ Quadrant II """
    t_q2 = cyclic_extraction(inv[2], inv[1], t)
    xe_q2, ye_q2, xc_q2, yc_q2, idx_q2, idy_q2, idx_lin_q2 = get_coordinates_and_indices(x_c, y_c, a, b, theta, x_grid, y_grid, t_q2)

    itg_q2 = (integral_quadratic_equation(f_1, f_2, f_3, f_4, f_5, f_6, xe_q2[1:], 0) -
              integral_quadratic_equation(f_1, f_2, f_3, f_4, f_5, f_6, xe_q2[:-1], 0)).real

    box_height_q2 = y_grid[idy_q2]
    box_height_q2[-1] = ye_q2[-1]
    # box_height_q2[box_height_q2 < ye_q2[-1]] = ye_q2[-1]

    box_q2 = box_height_q2 * np.diff(xe_q2)

    res[idx_q2, idy_q2] += box_q2 - itg_q2  # technically the other way around, but integrating from right to left

    # Add the rectangular shapes within the pixels
    right_not_on_x_grid_q2 = ~ np.isin(np.round(xe_q2[:-1], decimals=12), np.round(x_grid, decimals=12), assume_unique=True)
    right_not_on_x_grid_q2[0] = False

    # Index of the grid to the right of it
    idx_g_q2 = np.searchsorted(x_grid, xe_q2, side='right')
    x1_q2 = x_grid[idx_g_q2[:-1]]
    # x1_q2[x1_q2 > xe_q2[0]] = xe_q2[0]
    x0_q2 = xe_q2[:-1]
    dx_q2 = x1_q2 - x0_q2

    y0_q2 = y_grid[idy_q2]
    y0_q2[-1] = ye_q2[-1]
    # y0_q2[y0_q2 < ye_q2[-1]] = ye_q2[-1]
    y1_q2 = y_grid[idy_q2 + 1]

    dy_q2 = y1_q2 - y0_q2

    res[idx_q2[right_not_on_x_grid_q2], idy_q2[right_not_on_x_grid_q2]] += (dx_q2 * dy_q2)[right_not_on_x_grid_q2]

    """ Quadrant III """
    t_q3 = cyclic_extraction(inv[1], inv[3], t)
    xe_q3, ye_q3, xc_q3, yc_q3, idx_q3, idy_q3, idx_lin_q3 = get_coordinates_and_indices(x_c, y_c, a, b, theta, x_grid, y_grid, t_q3)

    itg_q3 = (integral_quadratic_equation(f_1, f_2, f_3, f_4, f_5, f_6, xe_q3[1:], 1) -
              integral_quadratic_equation(f_1, f_2, f_3, f_4, f_5, f_6, xe_q3[:-1], 1)).real

    box_height_q3 = y_grid[idy_q3 + 1]
    box_height_q3[0] = ye_q3[0]
    # box_height_q3[box_height_q3 > ye_q3[0]] = ye_q3[0])

    box_q3 = box_height_q3 * np.diff(xe_q3)

    res[idx_q3, idy_q3] += box_q3 - itg_q3

    # Add the rectangular shapes within the pixels
    right_not_on_x_grid_q3 = ~ np.isin(np.round(xe_q3[1:], decimals=12), np.round(x_grid, decimals=12), assume_unique=True)
    right_not_on_x_grid_q3[-1] = False

    # Index of the grid to the right of it
    idx_g_q3 = np.searchsorted(x_grid, xe_q3, side='right')

    x1_q3 = x_grid[idx_g_q3[1:]]
    # x1_q3[x1_q3 > xe_q3[-1]] = xe_q3[-1]
    x0_q3 = xe_q3[1:]
    dx_q3 = x1_q3 - x0_q3

    y0_q3 = y_grid[idy_q3 + 1]
    y0_q3[0] = ye_q3[0]
    # y0_q3[y0_q3 > ye_q3[0]] = ye_q3[0]
    y1_q3 = y_grid[idy_q3]
    dy_q3 = y0_q3 - y1_q3

    res[idx_q3[right_not_on_x_grid_q3], idy_q3[right_not_on_x_grid_q3]] += (dx_q3 * dy_q3)[right_not_on_x_grid_q3]

    """ Quadrant IV """
    t_q4 = cyclic_extraction(inv[3], inv[0], t)
    xe_q4, ye_q4, xc_q4, yc_q4, idx_q4, idy_q4, idx_lin_q4 = get_coordinates_and_indices(x_c, y_c, a, b, theta, x_grid, y_grid, t_q4)

    itg_q4 = (integral_quadratic_equation(f_1, f_2, f_3, f_4, f_5, f_6, xe_q4[1:], 1) -
              integral_quadratic_equation(f_1, f_2, f_3, f_4, f_5, f_6, xe_q4[:-1], 1)).real
    #
    box_height_q4 = y_grid[idy_q4 + 1]
    box_height_q4[-1] = ye_q4[-1]
    # box_height_q4[box_height_q4 > ye_q4[-1]] = ye_q4[-1]

    box_q4 = box_height_q4 * np.diff(xe_q4)

    res[idx_q4, idy_q4] += box_q4 - itg_q4

    # Add the rectangular shapes within the pixels
    left_not_on_x_grid_q4 = ~ np.isin(np.round(xe_q4[:-1], decimals=12), np.round(x_grid, decimals=12), assume_unique=True)
    left_not_on_x_grid_q4[0] = False

    # Index of the grid to the right of it
    idx_g_q4 = np.searchsorted(x_grid, xe_q4, side='right')

    x0_q4 = x_grid[idx_g_q4[:-1] - 1]
    # x0_q4[x0_q4 < xe_q4[0]] = xe_q4[0]
    x1_q4 = xe_q4[:-1]
    dx_q4 = x1_q4 - x0_q4

    y0_q4 = y_grid[idy_q4]
    y1_q4 = y_grid[idy_q4 + 1]
    y1_q4[-1] = ye_q4[-1]
    # y1_q4[y1_q4 > ye_q4[-1]] = ye_q4[-1]
    dy_q4 = y1_q4 - y0_q4

    res[idx_q4[left_not_on_x_grid_q4], idy_q4[left_not_on_x_grid_q4]] += (dx_q4 * dy_q4)[left_not_on_x_grid_q4]

    """"""
    res, pixel_area = fill_closed_domain(x_grid, y_grid, res)

    """"""
    idx_lin_sg, counts_sg = np.unique(np.concatenate((idx_lin_q1, idx_lin_q2, idx_lin_q3, idx_lin_q4)), return_counts=True)
    counts_ex = np.bincount(idx_lin_ex, minlength=(x_grid.size - 1) * (y_grid.size - 1))[idx_lin_sg]

    # If a pixel has segments from multiple quadrants but no extrema
    idx_dup_2, idy_dup_2 = np.unravel_index(idx_lin_sg[counts_sg - counts_ex > 1], (x_grid.size - 1, y_grid.size - 1))
    res[idx_dup_2, idy_dup_2] -= pixel_area[idx_dup_2, idy_dup_2]

    # If a pixel has segments from 3 quadrants
    aa = idx_lin_sg[(counts_sg == 3) & (counts_ex == 2)]
    idx_dup_3, idy_dup_3 = np.unravel_index(aa, (x_grid.size - 1, y_grid.size - 1))
    if aa.size > 0:
        # In the 1st quadrant
        if (xe_q1.size == 2) & (~ np.all(left_not_on_x_grid_q1)):
            print((xe_q2[0] - xe_q2[1]) * (ye_q4[-1] - ye_q4[-2]))
            res[idx_dup_3, idy_dup_3] -= (xe_q2[0] - xe_q2[1]) * (ye_q4[-1] - ye_q4[-2])

        # In the 2nd quadrant
        if (xe_q2.size == 2) & (~ np.all(right_not_on_x_grid_q2)):
            print((xe_q1[-2] - xe_q1[-1]) * (ye_q3[0] - ye_q3[1]))
            res[idx_dup_3, idy_dup_3] -= (xe_q1[-2] - xe_q1[-1]) * (ye_q3[0] - ye_q3[1])

        # In the 3rd quadrant
        if (xe_q3.size == 2) & (~ np.all(right_not_on_x_grid_q3)):
            print((ye_q2[-2] - ye_q2[-1]) * (xe_q4[1] - xe_q4[0]))
            res[idx_dup_3, idy_dup_3] -= (ye_q2[-2] - ye_q2[-1]) * (xe_q4[1] - xe_q4[0])

        # In the 4th quadrant
        if (xe_q4.size == 2) & (~ np.all(left_not_on_x_grid_q4)):
            print((ye_q1[1] - ye_q1[0]) * (xe_q3[-1] - xe_q3[-2]))
            res[idx_dup_3, idy_dup_3] -= (ye_q1[1] - ye_q1[0]) * (xe_q3[-1] - xe_q3[-2])

    # res[res > 1] -= 1
    res[res < 0] += 1

    area_difference = np.sum(res) - np.pi * a * b

    if (np.abs(area_difference) > 1e-11) and True:
        print(area_difference)
        x_s, y_s = t_parametrization(x_c, y_c, a, b, theta, np.linspace(0, 2 * np.pi, 1000))

        fig, ax = plt.subplots()
        ax.imshow(res.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
        ax.plot(x_s, y_s, color='k')
        ax.plot(xe_q1, ye_q1, '*')
        ax.plot(xe_q2, ye_q2, 'x')
        ax.plot(xe_q3, ye_q3, 'o', markerfacecolor='none')
        ax.plot(xe_q4, ye_q4, 's', markerfacecolor='none')

        ax.set_xticks(x_grid)
        ax.set_yticks(y_grid)
        ax.grid()
        ax.set_xlim(x_c - a - b, x_c + a + b)
        ax.set_ylim(y_c - a - b, y_c + a + b)
        ax.set_aspect(1)
        plt.show()

        # fig, ax = plt.subplots()
        # ax.plot(x_s, y_s, color='k')
        # ax.plot(xe_q1, ye_q1, '*')
        # ax.plot(xe_q2, ye_q2, 'x')
        # ax.plot(xe_q3, ye_q3, 'o', markerfacecolor='none')
        # ax.plot(xe_q4, ye_q4, 's', markerfacecolor='none')
        #
        # add_fill(ax, xe_q1, ye_q1, box_height_q1, 'tab:blue')
        # add_fill(ax, xe_q2, ye_q2, box_height_q2, 'tab:orange')
        # add_fill(ax, xe_q3, ye_q3, box_height_q3, 'tab:green')
        # add_fill(ax, xe_q4, ye_q4, box_height_q4, 'tab:red')
        #
        # add_rects(ax, x0_q1, x1_q1, y0_q1, y1_q1, left_not_on_x_grid_q1, 'tab:blue')
        # add_rects(ax, x0_q2, x1_q2, y0_q2, y1_q2, right_not_on_x_grid_q2, 'tab:orange')
        # add_rects(ax, x0_q3, x1_q3, y0_q3, y1_q3, right_not_on_x_grid_q3, 'tab:green')
        # add_rects(ax, x0_q4, x1_q4, y0_q4, y1_q4, left_not_on_x_grid_q4, 'tab:red')
        #
        # ax.set_xticks(x_plane)
        # ax.set_yticks(y_plane)
        # ax.grid()
        # ax.set_xlim(x_c - a - b, x_c + a + b)
        # ax.set_ylim(y_c - a - b, y_c + a + b)
        # # ax.set_aspect(1)
        # plt.show()

    return res


def add_fill(ax, x, y, offset, color):
    for ii in range(offset.size):
        ax.fill_between(x[ii:ii+2], y[ii:ii+2], y2=offset[ii], alpha=0.75, color=color)
    return 0


def add_rects(ax, x_0, x_1, y_0, y_1, valid, color):
    x_0 = x_0[valid]
    x_1 = x_1[valid]
    y_0 = y_0[valid]
    y_1 = y_1[valid]

    for ii in range(x_0.size):
        ax.add_patch(Rectangle((x_0[ii], y_0[ii]), x_1[ii] - x_0[ii], y_1[ii] - y_0[ii], alpha=0.75, hatch='/', facecolor='none', edgecolor=color))

    return


if __name__ == '__main__':
    # main()
    # main2()
    main3()
