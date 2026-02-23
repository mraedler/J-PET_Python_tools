"""
Write the GATE instructions to build the Derenzo phantom from
Am J Nucl Med Mol Imaging 2016;6(3):199-204

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


# Auxiliary functions


def main():
    # get_triangles_parameters(visualize=True)
    x, y, r, activities = get_derenzo_parameters(scaling_factor=3., visualize=True)

    # Additional parameters
    # z_shift = -815.0  # mm
    # z_shift = 755.0  # mm (inside the brain insert)
    z_shift = -755.0  # mm (on the opposite side of the brian insert)
    length = 50.0  # mm

    add_non_collinearity = True

    colors = np.array(['white', 'green', 'blue', 'cyan', 'magenta', 'yellow'])

    # mac_file = open('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Sources/Derenzo_Cox.mac', 'w')
    # mac_file = open('/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/Sources/Derenzo_Cox_3.mac', 'w')
    # mac_file = open('/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/Sources/Derenzo_Cox_3_outside.mac', 'w')
    # mac_file = open('/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/Sources/Derenzo_Cox_3_non_collinearity.mac', 'w')
    mac_file = open('/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/Sources/Derenzo_Cox_3_outside_non_collinearity.mac', 'w')
    mac_file.write('#=====================================================\n'
                   '#   PYTHON GENERATED GATE CODE FOR THE CONSTRUCTION\n'
                   '#   OF THE DERENZO PHANTOM PUBLISHED IN\n'
                   '#   Am J Nucl Med Mol Imaging 2016;6(3):199-204\n'
                   '#=====================================================\n')
    for ii in range(len(x)):
        mac_file.write('\n# Segment %d\n###########\n' % ii)
        for jj in range(x[ii].size):
            mac_file.write('\n# Rod %d\n' % jj)
            add_cylinder_gate(mac_file, ii, jj, x[ii][jj], y[ii][jj], z_shift, r[ii][jj], length, activities[ii], colors[ii], add_non_collinearity)

    mac_file.close()

    return 0


def get_triangles_parameters(return_valleys=False, visualize=False):
    # Parameteres
    idx = np.array([5])
    distances = np.array([10.0, 8.0, 6.0, 5.0, 4.0, 3.2])[idx]  # mm
    angles = np.array([0.])

    # n_rows = np.array([3, 4, 5, 6, 7, 8])[idx]
    n_rows = np.array([9, 12, 16, 20, 25, 31])[idx]

    # shifts = np.array([1, 1.5, 2, 2.5, 3, 3.5])[idx]
    shifts = np.array([5, 5.5, 8, 10.5, 13, 15.5])[idx]

    # center_offset = - distances * np.sqrt(3) / 2 * (n_rows - 1) / 2  # mm
    # center_offset = - distances * np.sqrt(3) / 2 * np.ceil(n_rows / 2)  # mm
    center_offset = - distances * np.sqrt(3) / 2 * shifts  # mm
    radii = distances / 4

    #
    x_peaks, y_peaks, r_peaks, _, _ = construct_phantom(n_rows, distances, radii, angles, center_offset)

    # Regroup the data into lines
    x_peaks_regrouped = regroup_into_lines(x_peaks, n_rows)
    y_peaks_regrouped = regroup_into_lines(y_peaks, n_rows)
    r_peaks_regrouped = regroup_into_lines(r_peaks, n_rows)

    # x_valleys_regrouped, y_valleys_regrouped, r_valleys_regrouped = get_valleys(x_peaks_regrouped, y_peaks_regrouped, r_peaks_regrouped)
    x_valleys_regrouped, y_valleys_regrouped, r_valleys_regrouped = get_valleys_parzych(x_peaks_regrouped, y_peaks_regrouped, r_peaks_regrouped)

    x_valleys = undo_regroup_into_lines(x_valleys_regrouped)
    y_valleys = undo_regroup_into_lines(y_valleys_regrouped)
    r_valleys = undo_regroup_into_lines(r_valleys_regrouped)

    if visualize:
        show_phantom(x_peaks, y_peaks, r_peaks)
        show_phantom(x_valleys, y_valleys, r_valleys)

    if return_valleys:
        return x_peaks, y_peaks, r_peaks, x_valleys, y_valleys, r_valleys
    else:
        return x_peaks, y_peaks, r_peaks


def get_derenzo_parameters(scaling_factor=3., return_valleys=False, visualize=False):
    # Original parameters
    n_rows = np.array([3, 4, 5, 6, 7, 8])
    distances = np.array([5.0, 4.0, 3.0, 2.5, 2.0, 1.6])  # mm
    center_offset = 7.5  # mm

    # Scale the size
    distances *= scaling_factor
    center_offset *= scaling_factor

    radii = distances / 4.
    angles = np.array([60., 120., 180., 240., 300., 360.])

    # Outer edges
    outer_edges = center_offset + (n_rows - 1) * distances * np.sqrt(3) / 2 + radii
    # print(outer_edges)

    x_peaks, y_peaks, r_peaks, x_out, y_out = construct_phantom(n_rows, distances, radii, angles, center_offset)

    # print(np.sqrt(np.sum(x_out, axis=1) ** 2 + np.sum(y_out, axis=1) ** 2) / 2 + radii)
    # np.save(sys.path[0] + '/Derenzo_pixelated/x_out.npy', x_out)
    # np.save(sys.path[0] + '/Derenzo_pixelated/y_out.npy', y_out)
    # np.save(sys.path[0] + '/Derenzo_pixelated/radii.npy', radii)

    # Regroup the data into lines
    x_peaks_regrouped = regroup_into_lines(x_peaks, n_rows)
    y_peaks_regrouped = regroup_into_lines(y_peaks, n_rows)
    r_peaks_regrouped = regroup_into_lines(r_peaks, n_rows)

    # x_valleys_regrouped, y_valleys_regrouped, r_valleys_regrouped = get_valleys(x_peaks_regrouped, y_peaks_regrouped, r_peaks_regrouped)
    x_valleys_regrouped, y_valleys_regrouped, r_valleys_regrouped = get_valleys_parzych(x_peaks_regrouped, y_peaks_regrouped, r_peaks_regrouped)

    x_valleys = undo_regroup_into_lines(x_valleys_regrouped)
    y_valleys = undo_regroup_into_lines(y_valleys_regrouped)
    r_valleys = undo_regroup_into_lines(r_valleys_regrouped)
    length = [group.size for group in r_valleys]

    if visualize:
        show_phantom(x_peaks, y_peaks, r_peaks)
        show_phantom(x_valleys, y_valleys, r_valleys)

    # Activity
    activity_max = 4e4  # Bq
    activities = radii ** 2 / radii[0] ** 2 * activity_max  # Bq
    print('Total activity: %1.2f MBq' % (np.sum(activities * n_rows * (n_rows + 1) / 2) / 1e6))

    if return_valleys:
        return x_peaks, y_peaks, r_peaks, x_valleys, y_valleys, r_valleys
    else:
        return x_peaks, y_peaks, r_peaks, activities


def construct_phantom(n_rows, distances, radii, angles, offset):
    x, y, r = [], [], []
    x_out, y_out = [], []
    for ii in range(n_rows.size):
        x_seg, y_seg = construct_segment(n_rows[ii], distances[ii])
        x_seg += 1e-10
        y_seg += offset

        alpha = -angles[ii] / 360 * 2 * np.pi
        x_rot = x_seg * np.cos(alpha) - y_seg * np.sin(alpha)
        y_rot = x_seg * np.sin(alpha) + y_seg * np.cos(alpha)

        x_out.append([x_rot[-1], x_rot[-n_rows[ii]]])
        y_out.append([y_rot[-1], y_rot[-n_rows[ii]]])

        x.append(x_rot)
        y.append(y_rot)
        r.append(radii[ii] * np.ones(x_rot.shape))

    x_out = np.array(x_out)
    y_out = np.array(y_out)

    return x, y, r, x_out, y_out


def construct_segment(n_row, distance):
    # Ratio between side length and height in an equilateral triangle
    ratio = np.sqrt(3) / 2

    x, y = [], []
    for nn in range(n_row):
        x.append(uniform_centered_coordinates(nn + 1) * distance)
        y.append(nn * np.ones((nn + 1,)) * ratio * distance)

    x = np.concatenate(x)
    y = np.concatenate(y)

    # Check if the number of elements corresponds to the triangle number
    tril = int(n_row * (n_row + 1) / 2)
    if x.size != tril or y.size != tril:
        print('Something went wrong!')

    return x, y


def uniform_centered_coordinates(n):
    return np.arange(n) - (n - 1) / 2


def show_phantom(x, y, r):
    fig, ax = plt.subplots()
    for ii in range(len(x)):
        for jj in range(x[ii].size):
            ax.add_artist(Circle((x[ii][jj], y[ii][jj]), r[ii][jj], alpha=0.75))

    mx = np.max(np.abs(np.concatenate(x + y))) * 1.1
    # mx = 20.
    ax.set_xlim(-mx, mx)
    ax.set_ylim(-mx, mx)
    ax.set_aspect('equal')
    plt.show()
    return 0


def regroup_into_lines(x, n_rows):
    if len(x) != n_rows.size:
        sys.exit('Error: len(x) != n_rows.size')

    x_regrouped = []
    for ii in range(n_rows.size):
        n = np.arange(n_rows[ii] + 1)
        tril = (n * (n + 1) / 2).astype(int)

        line_separated = []
        for jj in range(n_rows[ii]):
            line_separated.append(x[ii][tril[jj]:tril[jj + 1]])

        x_regrouped.append(line_separated)

    return x_regrouped


def get_valleys(x_peaks_regrouped, y_peaks_regrouped, r_peaks_regrouped):

    x_valleys_regrouped, y_valleys_regrouped, r_valleys_regrouped = [], [], []

    for ii in range(len(x_peaks_regrouped)):
        x_lines, y_lines, r_lines = [], [], []
        for jj in range(len(x_peaks_regrouped[ii]) - 1):
            x_lines.append((x_peaks_regrouped[ii][jj] + x_peaks_regrouped[ii][jj + 1][1:] + x_peaks_regrouped[ii][jj + 1][:-1]) / 3)
            y_lines.append((y_peaks_regrouped[ii][jj] + y_peaks_regrouped[ii][jj + 1][1:] + y_peaks_regrouped[ii][jj + 1][:-1]) / 3)
            r_lines.append((r_peaks_regrouped[ii][jj] + r_peaks_regrouped[ii][jj + 1][1:] + r_peaks_regrouped[ii][jj + 1][:-1]) / 3)

        x_valleys_regrouped.append(x_lines)
        y_valleys_regrouped.append(y_lines)
        r_valleys_regrouped.append(r_lines)

    return x_valleys_regrouped, y_valleys_regrouped, r_valleys_regrouped


def get_valleys_parzych(x_peaks_regrouped, y_peaks_regrouped, r_peaks_regrouped):

    x_valleys_regrouped, y_valleys_regrouped, r_valleys_regrouped = [], [], []

    for ii in range(len(x_peaks_regrouped)):
        x_lines, y_lines, r_lines = [], [], []
        for jj in range(len(x_peaks_regrouped[ii]) - 1):
            x_lines.append((x_peaks_regrouped[ii][jj] + x_peaks_regrouped[ii][jj + 1][:-1]) / 2)
            x_lines.append((x_peaks_regrouped[ii][jj] + x_peaks_regrouped[ii][jj + 1][1:]) / 2)
            x_lines.append((x_peaks_regrouped[ii][jj + 1][:-1] + x_peaks_regrouped[ii][jj + 1][1:]) / 2)

            y_lines.append((y_peaks_regrouped[ii][jj] + y_peaks_regrouped[ii][jj + 1][:-1]) / 2)
            y_lines.append((y_peaks_regrouped[ii][jj] + y_peaks_regrouped[ii][jj + 1][1:]) / 2)
            y_lines.append((y_peaks_regrouped[ii][jj + 1][:-1] + y_peaks_regrouped[ii][jj + 1][1:]) / 2)

            r_lines.append((r_peaks_regrouped[ii][jj] + r_peaks_regrouped[ii][jj + 1][:-1]) / 2)
            r_lines.append((r_peaks_regrouped[ii][jj] + r_peaks_regrouped[ii][jj + 1][1:]) / 2)
            r_lines.append((r_peaks_regrouped[ii][jj + 1][:-1] + r_peaks_regrouped[ii][jj + 1][1:]) / 2)

        x_valleys_regrouped.append(x_lines)
        y_valleys_regrouped.append(y_lines)
        r_valleys_regrouped.append(r_lines)

    return x_valleys_regrouped, y_valleys_regrouped, r_valleys_regrouped


def undo_regroup_into_lines(lst):
    lst_red = []
    for ii in range(len(lst)):
        lst_red.append(np.concatenate(lst[ii]))
    return lst_red


def add_cylinder_gate(mac_file, sec_idx, rod_idx, x, y, z, r, h, activity, color, add_non_collinearity):
    source_name = 'rod_%d_%d' % (sec_idx, rod_idx)
    mac_file.write('/gate/source/addSource %s\n' % source_name)
    mac_file.write('/gate/source/%s/setType backtoback\n' % source_name)

    if add_non_collinearity:
        mac_file.write('/gate/source/%s/setAccolinearityFlag True\n' % source_name)
        mac_file.write('/gate/source/%s/setAccoValue 0.5 deg\n' % source_name)

    mac_file.write('/gate/source/%s/gps/type Volume\n' % source_name)
    mac_file.write('/gate/source/%s/gps/shape Cylinder\n' % source_name)
    mac_file.write('/gate/source/%s/gps/radius %1.3f mm\n' % (source_name, r))
    mac_file.write('/gate/source/%s/gps/halfz %1.3f mm\n' % (source_name, h / 2))
    mac_file.write('/gate/source/%s/gps/centre %1.3f %1.3f %1.3f mm\n' % (source_name, x, y, z))
    mac_file.write('/gate/source/%s/gps/particle gamma\n' % source_name)
    mac_file.write('/gate/source/%s/gps/energytype Mono\n' % source_name)
    mac_file.write('/gate/source/%s/gps/monoenergy 511 keV\n' % source_name)
    mac_file.write('/gate/source/%s/gps/angtype iso\n' % source_name)
    mac_file.write('/gate/source/%s/setActivity %1.3f Bq\n' % (source_name, activity))
    mac_file.write('/gate/source/%s/visualize 100 %s 5\n' % (source_name, color))
    return


if __name__ == "__main__":
    main()
