"""
Generate the binary LUT file for CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    # In CASToR, the lookup table (LUT) has a set floating point precision (FLTNB: float n bits) called FLTNBLUT
    # We need to choose the same here, that was set during the compilation of CASToR
    fltnblut = np.float32

    # For consistency, we also unify the integer type
    intnblut = np.int32

    # # Allocate the parameter lists
    # radius, axial_offset, delta_rings = [], [], []  # Global placement
    # n_rings, n_angles, n_lat, n_zed = [], [], [], []  # Number of elements
    # delta_rad, delta_lat, delta_zed = [], [], []  # Crystal spacing
    # depth, transaxial, axial = [], [], []  # Crystal dimensions

    # Todo: automize this, i.e. read the geometry parameters from the GATE macro
    """Scanner parameters"""
    # scanner_name = 'TB_J-PET_7th_gen'
    # description = 'Two rings with each 33 cm, and three rings with each 60 cm axial length.'

    # Brain insert
    (n_angles, radius,
     n_rings, delta_rings,
     n_crystal_lat, delta_crystal_lat,
     n_crystal_zed, delta_crystal_zed,
     depth, transaxial, axial) = get_siemens_biograph_40_mCT_parameters(fltnblut, intnblut)

    scanner_name = 'Siemens_Biograph_40_mCT'
    description = 'Using the geometry from https://github.com/mraedler/Siemens_Biograph_mCT'

    generate_lut(fltnblut, intnblut,
                 n_angles, radius,
                 n_rings, delta_rings,
                 n_crystal_lat, delta_crystal_lat,
                 n_crystal_zed, delta_crystal_zed,
                 depth, transaxial, axial,
                 scanner_name, description)

    return 0


def get_siemens_biograph_40_mCT_parameters(fltnblut, intnblut):
    # blockID (usually rsectorID)
    n_angles = intnblut(48)
    radius = fltnblut(438.)  # [mm]

    # blockID (usually moduleID or crystalID)
    n_rings = intnblut(4)
    delta_rings = fltnblut(56.)  # [mm]

    # Crystal
    n_crystal_lat = intnblut(13)
    delta_crystal_lat = fltnblut(4.)  # [mm]
    n_crystal_zed = intnblut(13)
    delta_crystal_zed = fltnblut(4.)  # [mm]

    # Size
    depth = fltnblut(20.)  # [mm]
    transaxial = fltnblut(3.98)  # [mm]
    axial = fltnblut(3.98)  # [mm]

    return (n_angles, radius,
            n_rings, delta_rings,
            n_crystal_lat, delta_crystal_lat,
            n_crystal_zed, delta_crystal_zed,
            depth, transaxial, axial)


def generate_lut(fltnblut, intnblut,
                 n_angles, radius,
                 n_rings, delta_rings,
                 n_crystal_lat, delta_crystal_lat,
                 n_crystal_zed, delta_crystal_zed,
                 depth, transaxial, axial,
                 scanner_name, description):
    #
    n_elements = n_angles * n_rings * n_crystal_lat * n_crystal_zed

    # block
    ring_centers_zed = symmetric_grid(n_rings, dtype=fltnblut) * delta_rings

    #
    crystal_centers_zed = symmetric_grid(n_crystal_zed, dtype=fltnblut) * delta_crystal_zed
    crystal_centers_lat = symmetric_grid(n_crystal_lat, dtype=fltnblut) * delta_crystal_lat

    # print(ring_centers_zed[1] + crystal_centers_zed)
    # sys.exit()

    # 4 blocks
    z_r, z_c, y_c = np.meshgrid(ring_centers_zed, crystal_centers_zed, crystal_centers_lat, indexing='ij')

    # # Visualization
    # fig, ax = plt.subplots()
    # ax.scatter(z_r + z_c, y_c, alpha=0.1)
    # # ax.scatter(z_c, y_c, alpha=0.1)
    # ax.set_aspect(1)
    # plt.show()

    zed = (z_r + z_c).ravel()
    lat = y_c.ravel()
    rad = radius * np.ones(zed.shape)
    phi_0 = -4.28572  # [deg]
    phi = (np.arange(n_angles, dtype=fltnblut) / n_angles + phi_0 / 360.) * 2 * np.pi

    pos_x, pos_y, pos_z = [], [], []
    or_vx, or_vy, or_vz = [], [], []

    for ii in range(n_angles):
        pos_x.append(rad * np.cos(phi[ii]) - lat * np.sin(phi[ii]))
        pos_y.append(rad * np.sin(phi[ii]) + lat * np.cos(phi[ii]))
        pos_z.append(zed)

        or_vx.append(np.cos(phi[ii]) * np.ones(rad.shape, dtype=fltnblut))
        or_vy.append(np.sin(phi[ii]) * np.ones(rad.shape, dtype=fltnblut))
        or_vz.append(np.zeros(rad.shape, dtype=fltnblut))

    pos_x, pos_y, pos_z = np.concatenate(pos_x), np.concatenate(pos_y), np.concatenate(pos_z)
    or_vx, or_vy, or_vz = np.concatenate(or_vx), np.concatenate(or_vy), np.concatenate(or_vz)

    # Check the normalization (should be normalized to one)
    print(or_vx ** 2 + or_vy ** 2 + or_vz ** 2)

    # Allocate a numpy structured array
    d_type = np.dtype([('Posx', fltnblut), ('Posy', fltnblut), ('Posz', fltnblut),
                       ('OrVx', fltnblut), ('OrVy', fltnblut), ('OrVz', fltnblut)])
    lut = np.empty(n_elements, dtype=d_type)  # Use np.empty(n, dtype=dtype)

    # Write the data
    lut['Posx'], lut['Posy'], lut['Posz'] = pos_x, pos_y, pos_z
    lut['OrVx'], lut['OrVy'], lut['OrVz'] = or_vx, or_vy, or_vz

    # Visualization
    visualize_lut_entries(lut, np.array(n_elements))

    # Save as binary file
    open(scanner_name + '.lut', 'wb').write(lut.tobytes())

    # Write the header file
    generate_hscan_file(scanner_name, description, lut.size, 1, np.array(n_elements, ndmin=1),
                        np.array(depth, ndmin=1), np.array(transaxial, ndmin=1), np.array(axial, ndmin=1))

    return 0


def symmetric_grid(n, dtype):
    return np.arange(n, dtype=dtype) - (n - 1) / 2


def generate_hscan_file(scanner_name, description, n_elements, n_layers, n_crystals_per_layer, depth, transaxial, axial):
    mandatory_fields = {'modality': 'PET',
                        'scanner name': scanner_name,
                        'description': description,
                        'number of elements': str(n_elements),
                        'number of layers': str(np.sum(n_layers)),
                        'voxels number transaxial': '256',
                        'voxels number axial': '256',
                        'field of view transaxial': '256.0  # [mm]',
                        'field of view axial': '256.0  # [mm]',
                        '\n# The following entries depend on the number of layers': '',
                        'number of crystals in layer': ",".join(map(str, n_crystals_per_layer)),
                        'crystals size depth': ",".join(map(str, depth)) + '  # [mm]',
                        'crystals size transaxial': ",".join(map(str, transaxial)) + '  # [mm]',
                        'crystals size axial': ",".join(map(str, axial)) + '  # [mm]',
                        'mean depth of interaction': ",".join(map(str, -np.ones(depth.size, dtype=int))),
                        'min angle difference': ",".join(map(str, np.zeros(depth.size, dtype=int)))}

    cdh_file = open(scanner_name + '.hscan', 'w')
    for key, value in mandatory_fields.items():
        cdh_file.write(key + ': ' + value + '\n')
    cdh_file.close()

    return 0


def visualize_lut_entries(lut, n_elements_per_layer):
    plt.rcParams.update({'font.size': 16})
    colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors = np.repeat(colors, 2)  #
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 4), width_ratios=(1, 2.5), sharey=True)

    s = np.insert(np.cumsum(n_elements_per_layer), 0, 0)
    for ii in range(n_elements_per_layer.size):
        pos_x, pos_y, pos_z = lut['Posx'][s[ii]:s[ii + 1]], lut['Posy'][s[ii]:s[ii + 1]], lut['Posz'][s[ii]:s[ii + 1]]
        or_vx, or_vy, or_vz = lut['OrVx'][s[ii]:s[ii + 1]], lut['OrVy'][s[ii]:s[ii + 1]], lut['OrVz'][s[ii]:s[ii + 1]]

        z_slices, idx = np.unique(pos_z, return_inverse=True)

        # In the x-y plane, only plot the first slice
        ax0.quiver(pos_x[idx == 0], pos_y[idx == 0],
                   or_vx[idx == 0], or_vy[idx == 0], angles='xy', scale_units='xy', scale=1 / 20, color=colors[ii])

        # In the y-z plane, plot every 10-th slice
        for jj in range(0, z_slices.size, 1):
            ax1.quiver(pos_z[idx == jj], pos_y[idx == jj],
                       or_vz[idx == jj], or_vy[idx == jj], angles='xy', scale_units='xy', scale=1 / 20, color=colors[ii])

    ax0.set_aspect(1)
    ax1.set_aspect(1)

    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_ylabel(r'$y$ [mm]')

    ax1.set_xlabel(r'$z$ [mm]')
    # ax1.set_ylabel(r'$y$ [mm]')

    plt.show()

    return 0


if __name__ == "__main__":
    main()
