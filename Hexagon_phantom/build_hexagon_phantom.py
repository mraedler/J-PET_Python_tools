"""
Build a hexagonal phantom to examine the radial resolution

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

# Auxiliary functions
from Derenzo_phantom.build_derenzo_phantom import add_cylinder_gate


def main():
    radii = 2.
    x_rods, y_rods = get_hexagon_parameters(radii=radii, n_layers=25, scanner_radius=160., visualize=False)
    # x_rods, y_rods = get_hexagon_parameters(radii=radii, n_layers=20, scanner_radius=160., visualize=True)

    pp = [x_rods[ii].size for ii in range(len(x_rods))]
    print(pp)
    print(np.sum(pp) * 800 / 1e6)

    sys.exit()

    np.save(sys.path[0] + '/Hexagon/x_rods.npy', np.array(x_rods, dtype=object))
    np.save(sys.path[0] + '/Hexagon/y_rods.npy', np.array(y_rods, dtype=object))
    np.save(sys.path[0] + '/Hexagon/radii.npy', np.array(radii))
    sys.exit()

    n = [x_rods[ii].size for ii in range(len(x_rods))]
    activity = np.round(1e6 / np.sum(n))
    activity = 800  # Bq
    print('Total activity: %1.3f MBq' % (np.sum(n) * activity / 1e6))

    # Additional parameters
    z_shift = -815.0  # mm
    length = 50.0  # mm

    colors = np.array(['white', 'green', 'blue', 'cyan', 'magenta', 'yellow'])

    mac_file = open('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Sources/Hexagon.mac', 'w')
    mac_file.write('#=====================================================\n'
                   '#   PYTHON GENERATED GATE CODE FOR THE CONSTRUCTION OF\n'
                   '#   A PHANTOM OF RODS ARRANGED IN A HEXAGONAL PATTERN\n'
                   '#=====================================================\n')

    for ii in range(len(x_rods)):
        mac_file.write('\n# Hexagon %d\n###########\n' % ii)
        for jj in range(x_rods[ii].size):
            mac_file.write('\n# Rod %d\n' % jj)
            add_cylinder_gate(mac_file, ii, jj, x_rods[ii][jj], y_rods[ii][jj], z_shift, radii, length, activity, colors[ii % 6])

    mac_file.close()


    return 0


def get_hexagon_parameters(radii=1., n_layers=3, scanner_radius=10., visualize=False):
    # Get 6 unit vectors pointing at the corners of hexagons
    theta = np.arange(0, 360, 60, dtype=float)[:, np.newaxis] / 360 * 2 * np.pi
    positions = np.hstack((np.cos(theta), np.sin(theta)))
    velocities = np.diff(positions, axis=0, append=positions[0:1, :])

    # fig, ax = plt.subplots()
    # ax.scatter(positions[:, 0], positions[:, 1])
    # ax.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1], scale_units='xy', angles='xy', scale=2)
    # ax.set_aspect(1)
    # plt.show()

    x_rods, y_rods = [np.array([0.])], [np.array([0.])]

    if visualize:
        fig, ax = plt.subplots()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax.add_patch(Circle((x_rods[0][0], y_rods[0][0]), radii, color=colors[0], fill=True))

    for ii in np.arange(1, n_layers):

        alpha = np.arange(ii, dtype=float)[np.newaxis, :] / ii
        x_rods_ii = ii * (positions[:, 0:1] + alpha * velocities[:, 0:1]).flatten(order='C') * 4 * radii
        y_rods_ii = ii * (positions[:, 1:] + alpha * velocities[:, 1:]).flatten(order='C') * 4 * radii
        within_scanner = (np.sqrt(x_rods_ii ** 2 + y_rods_ii ** 2) + radii) < scanner_radius

        x_rods_ii, y_rods_ii = x_rods_ii[within_scanner], y_rods_ii[within_scanner]

        # print(x_rods[ii].size)
        if x_rods_ii.size > 0:
            x_rods.append(x_rods_ii)
            y_rods.append(y_rods_ii)

        if visualize:
            for jj in range(x_rods_ii.size):
                ax.add_patch(Circle((x_rods[ii][jj], y_rods[ii][jj]), radii, color=colors[ii % len(colors)], fill=True))

    if visualize:
        ax.add_patch(Circle((0., 0.), scanner_radius, color='k', fill=False))

        mx = np.max(np.abs(np.concatenate(x_rods + y_rods))) + 2 * radii
        ax.set_xlim(-mx, mx)
        ax.set_ylim(-mx, mx)

        ax.set_aspect(1)
        plt.show()

    # r_rods = np.sqrt(np.concatenate(x_rods) ** 2 + np.concatenate(y_rods) ** 2)
    #
    # r_edges = np.linspace(0, scanner_radius, 16)
    # h, _ = np.histogram(r_rods, bins=r_edges)
    #
    # fig, ax = plt.subplots()
    # ax.stairs(h, edges=r_edges, color='k')
    # plt.show()

    return x_rods, y_rods


if __name__ == "__main__":
    main()
