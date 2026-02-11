"""
Pixelate a given contour

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import split
from shapely.validation import explain_validity
from matplotlib import pyplot as plt


def main():
    x_c, y_c, a, b, theta = 10., 0, 9.5, 9.5, 0

    n_contour = 1000
    t = np.arange(n_contour) / n_contour * 2 * np.pi
    # t = (np.arange(n_contour) / n_contour * 2 * np.pi + np.pi) % (2 * np.pi)
    x_contour, y_contour = t_parametrization(x_c, y_c, a, b, theta, t)

    x_grid = np.arange(-64, 64 + 1)
    y_grid = np.arange(-64, 64 + 1)

    img = pixelate_contour(x_grid, y_grid, x_contour, y_contour, show_figure=True)

    print(np.sum(img) - np.pi * a * b)

    return 0


def t_parametrization(x_c, y_c, a, b, theta, t):
    x = a * np.cos(theta) * np.cos(t) - b * np.sin(theta) * np.sin(t) + x_c
    y = a * np.sin(theta) * np.cos(t) + b * np.cos(theta) * np.sin(t) + y_c
    return x, y


def pixelate_contour(x_g, y_g, x_c, y_c, show_figure=False):
    """
    :param x_g: 1D array of grid edges along x
    :param y_g: 1D array of grid edges along y
    :param x_c: x-coordinates of the contour
    :param y_c: y-coordinates of the contour
    :param show_figure: bool
    :return:
    """
    img = np.zeros((x_g.size - 1, y_g.size - 1))

    # Add the first point to the end to close the contour
    x_cc = np.append(x_c, x_c[0])
    y_cc = np.append(y_c, y_c[0])

    # Determine the intersections with the grid

    # Turn the curve into a parametric curve
    t = np.arange(len(x_cc))

    # Get the t-values for horizontal and vertical intersections
    idx_x_grid, idx_t = np.where((x_g[:, None] - x_cc[None, :-1]) * (x_g[:, None] - x_cc[None, 1:]) <= 0)
    t_x = idx_t + (x_g[idx_x_grid] - x_cc[idx_t]) / (x_cc[idx_t + 1] - x_cc[idx_t])

    idx_y_grid, idx_t = np.where((y_g[:, None] - y_cc[None, :-1]) * (y_g[:, None] - y_cc[None, 1:]) <= 0)
    t_y = idx_t + (y_g[idx_y_grid] - y_cc[idx_t]) / (y_cc[idx_t + 1] - y_cc[idx_t])

    # t-values that correspond to the grid intersections and estimate the corresponding x- and y-coordinates
    t_its = np.sort(np.unique(np.round(np.hstack((t_x, t_y)), decimals=12)))
    # t_its = np.append(t_its, t_its[0])
    x_its = np.interp(t_its, t, x_cc)
    y_its = np.interp(t_its, t, y_cc)

    # Determine, mid-points and the corresponding indices
    x_mid = np.append(x_its, x_its[0])
    y_mid = np.append(y_its, y_its[0])
    x_mid = (x_mid[:-1] + x_mid[1:]) / 2
    y_mid = (y_mid[:-1] + y_mid[1:]) / 2

    idx_cx = np.digitize(x_mid, x_g) - 1
    idx_cy = np.digitize(y_mid, y_g) - 1

    idx_lin = np.ravel_multi_index((idx_cx, idx_cy), img.shape)
    idx_lin_unique = np.unique(idx_lin)
    if idx_lin.size - idx_lin_unique.size != 0:
        # print('Duplicated pixels found.')
        idx_cx, idx_cy = np.unravel_index(idx_lin_unique, img.shape)

    # Turn the entire contour into a shapely line string and polygon
    contour_string = LineString([(x_cc[ii], y_cc[ii]) for ii in range(x_cc.size)])
    contour_polygon = Polygon([(x_c[ii], y_c[ii]) for ii in range(x_c.size)])

    for ii in range(idx_cx.size):
        # Define square polygon
        square_polygon = Polygon([(x_g[idx_cx[ii]], y_g[idx_cy[ii]]),
                                  (x_g[idx_cx[ii] + 1], y_g[idx_cy[ii]]),
                                  (x_g[idx_cx[ii] + 1], y_g[idx_cy[ii] + 1]),
                                  (x_g[idx_cx[ii]], y_g[idx_cy[ii] + 1])])

        img[idx_cx[ii], idx_cy[ii]] = contour_polygon.intersection(square_polygon).area

        # # Split the square using the multi-segment line
        # split_polygons = split(square_polygon, contour_string).geoms
        #
        # fig, ax = plt.subplots()
        # ax.plot(*square_polygon.exterior.xy)
        #
        # for jj in range(len(split_polygons)):
        #     print(split_polygons[jj].area)
        #     # print(split_polygons[jj].overlaps(contour_polygon))
        #     ax.plot(*split_polygons[jj].exterior.xy, '--')
        # plt.show()
        # print()

    img, _ = fill_convex_closed_domain(x_g, y_g, img)

    if show_figure:
        fig, ax = plt.subplots()
        ax.imshow(img.T, origin='lower', extent=[x_g[0], x_g[-1], y_g[0], y_g[-1]])
        ax.plot(x_cc, y_cc)
        ax.plot(x_its, y_its, 'x')
        ax.set_xticks(x_g)
        ax.set_yticks(y_g)
        ax.grid()
        ax.set_xlim(np.min(x_c) - 1, np.max(x_c) + 1)
        ax.set_ylim(np.min(y_c) - 1, np.max(y_c) + 1)
        plt.show()

    return img


def fill_convex_closed_domain(x_grid, y_grid, img):
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


if __name__ == "__main__":
    main()
