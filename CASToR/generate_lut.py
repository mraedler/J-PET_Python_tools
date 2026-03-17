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

    # Allocate the parameter lists
    radius, axial_offset, delta_rings = [], [], []  # Global placement
    n_rings, n_angles, n_lat, n_zed = [], [], [], []  # Number of elements
    delta_rad, delta_lat, delta_zed = [], [], []  # Crystal spacing
    depth, transaxial, axial = [], [], []  # Crystal dimensions

    # Todo: automize this, i.e. read the geometry parameters from the GATE macro
    """Scanner parameters"""
    # scanner_name = 'TB_J-PET_7th_gen'
    # description = 'Two rings with each 33 cm, and three rings with each 60 cm axial length.'

    # Total body J-PET 7th generation
    (radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial) = get_total_body_j_pet_7th_gen_parameters(
        radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial, fltnblut, intnblut)

    # # Modular scanner
    # (radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial) = get_modular_scanner_parameters(
    #     radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial, fltnblut, intnblut)

    # # Brain insert
    # (radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial) = get_brain_insert_12_parameters_whr_4_18_1(
    #     radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial, fltnblut, intnblut)

    # Brain insert
    (radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial) = get_brain_insert_12_parameters_whr_6_30_1(
        radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial, fltnblut, intnblut)

    scanner_name = 'TB_J-PET_7th_gen_brain_insert_WHR_6_30_1_mm'
    description = ('Two rings with each 33 cm, three rings with each 60 cm axial length, and a brain insert. '
                   'Z binning reduced to 1 mm for more accurate blurring.')

    # scanner_name = 'Modular_scanner'
    # description = 'Add description.'

    generate_lut(radius, axial_offset, delta_rings,
                 n_rings, n_angles, n_lat, n_zed,
                 delta_rad, delta_lat, delta_zed,
                 depth, transaxial, axial,
                 fltnblut, scanner_name, description)

    return 0


def get_modular_scanner_parameters(radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed,
                                   delta_rad, delta_lat, delta_zed, depth, transaxial, axial, fltnblut, intnblut):
    radius.append(fltnblut(381.86))  # [mm]
    axial_offset.append(fltnblut(0.))  # [mm]
    delta_rings.append(fltnblut(0.))  # [mm]
    n_rings.append(intnblut(1))
    n_angles.append(intnblut(24))
    n_lat.append(np.array([13], dtype=intnblut))
    n_zed.append(np.array([200], dtype=intnblut))
    delta_rad.append(np.array([0.], dtype=fltnblut))  # [mm]
    delta_lat.append(np.array([7.], dtype=fltnblut))  # [mm]
    delta_zed.append(np.array([2.5], dtype=fltnblut))  # [mm]
    depth.append(np.array([24.], dtype=fltnblut))  # [mm]
    transaxial.append(np.array([6.], dtype=fltnblut))  # [mm]
    axial.append(np.array([2.5], dtype=fltnblut))  # [mm]

    return radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial


def get_total_body_j_pet_7th_gen_parameters(radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed,
                                            delta_rad, delta_lat, delta_zed, depth, transaxial, axial, fltnblut, intnblut):
    # 2 rings with each 33 cm length
    radius.append(fltnblut(446.599))  # [mm]
    axial_offset.append(fltnblut(930.))  # [mm]
    delta_rings.append(fltnblut(350.))  # [mm]
    n_rings.append(intnblut(2))
    n_angles.append(intnblut(24))
    n_lat.append(np.array([16, 16], dtype=intnblut))
    n_zed.append(np.array([330, 330], dtype=intnblut))
    delta_rad.append(np.array([-16.8, 16.8], dtype=fltnblut))  # [mm]
    delta_lat.append(np.array([6.6, 6.6], dtype=fltnblut))  # [mm]
    delta_zed.append(np.array([1., 1.], dtype=fltnblut))  # [mm]
    depth.append(np.array([30., 30.], dtype=fltnblut))  # [mm]
    transaxial.append(np.array([6., 6.], dtype=fltnblut))  # [mm]
    axial.append(np.array([1., 1.], dtype=fltnblut))  # [mm]

    # 3 rings with each 60 cm length
    radius.append(fltnblut(448.099))  # [mm]
    axial_offset.append(fltnblut(-350.))  # [mm]
    delta_rings.append(fltnblut(620.))  # [mm]
    n_rings.append(intnblut(3))
    n_angles.append(intnblut(24))
    n_lat.append(np.array([16, 16], dtype=intnblut))
    n_zed.append(np.array([600, 600], dtype=intnblut))
    delta_rad.append(np.array([-18.3, 18.3], dtype=fltnblut))  # [mm]
    delta_lat.append(np.array([6.6, 6.6], dtype=fltnblut))  # [mm]
    delta_zed.append(np.array([1., 1.], dtype=fltnblut))  # [mm]
    depth.append(np.array([30., 30.], dtype=fltnblut))  # [mm]
    transaxial.append(np.array([6., 6.], dtype=fltnblut))  # [mm]
    axial.append(np.array([1., 1.], dtype=fltnblut))  # [mm]

    return radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial


def get_brain_insert_10_parameters(radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed,
                                   delta_rad, delta_lat, delta_zed, depth, transaxial, axial, fltnblut, intnblut):
    radius.append(fltnblut(188.))  # [mm]
    axial_offset.append(fltnblut(755.))  # [mm]
    delta_rings.append(fltnblut(0.))  # [mm]
    n_rings.append(intnblut(1))
    n_angles.append(intnblut(10))
    n_lat.append(np.array([23, 23], dtype=intnblut))
    n_zed.append(np.array([110, 110], dtype=intnblut))
    delta_rad.append(np.array([-10.8, 10.8], dtype=fltnblut))  # [mm]
    delta_lat.append(np.array([4.5909090909, 4.5909090909], dtype=fltnblut))  # [mm]
    delta_zed.append(np.array([3., 3.], dtype=fltnblut))  # [mm]
    depth.append(np.array([18., 18.], dtype=fltnblut))  # [mm]
    transaxial.append(np.array([4., 4.], dtype=fltnblut))  # [mm]
    axial.append(np.array([3., 3.], dtype=fltnblut))  # [mm]

    return radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial


def get_brain_insert_12_parameters_whr_4_18_1(radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed,
                                              delta_rad, delta_lat, delta_zed, depth, transaxial, axial, fltnblut, intnblut):
    radius.append(fltnblut(164.6))  # [mm]
    axial_offset.append(fltnblut(755.))  # [mm]
    delta_rings.append(fltnblut(0.))  # [mm]
    n_rings.append(intnblut(1))
    n_angles.append(intnblut(12))
    n_lat.append(np.array([16, 16], dtype=intnblut))
    n_zed.append(np.array([330, 330], dtype=intnblut))
    delta_rad.append(np.array([-11.3, 11.3], dtype=fltnblut))  # [mm]
    delta_lat.append(np.array([4.6, 4.6], dtype=fltnblut))  # [mm]
    delta_zed.append(np.array([1., 1.], dtype=fltnblut))  # [mm]
    depth.append(np.array([18., 18.], dtype=fltnblut))  # [mm]
    transaxial.append(np.array([4., 4.], dtype=fltnblut))  # [mm]
    axial.append(np.array([1., 1.], dtype=fltnblut))  # [mm]

    return radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial


def get_brain_insert_12_parameters_whr_6_30_1(radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed,
                                              delta_rad, delta_lat, delta_zed, depth, transaxial, axial, fltnblut, intnblut):
    radius.append(fltnblut(176.1))  # [mm]
    axial_offset.append(fltnblut(755.))  # [mm]
    delta_rings.append(fltnblut(0.))  # [mm]
    n_rings.append(intnblut(1))
    n_angles.append(intnblut(12))
    n_lat.append(np.array([11, 11], dtype=intnblut))
    n_zed.append(np.array([330, 330], dtype=intnblut))
    delta_rad.append(np.array([-16.8, 16.8], dtype=fltnblut))  # [mm]
    delta_lat.append(np.array([6.6, 6.6], dtype=fltnblut))  # [mm]
    delta_zed.append(np.array([1., 1.], dtype=fltnblut))  # [mm]
    depth.append(np.array([30., 30.], dtype=fltnblut))  # [mm]
    transaxial.append(np.array([6., 6.], dtype=fltnblut))  # [mm]
    axial.append(np.array([1., 1.], dtype=fltnblut))  # [mm]

    return radius, axial_offset, delta_rings, n_rings, n_angles, n_lat, n_zed, delta_rad, delta_lat, delta_zed, depth, transaxial, axial


def generate_lut(radius, axial_offset, delta_rings,
                 n_rings, n_angles, n_lat, n_zed,
                 delta_rad, delta_lat, delta_zed,
                 depth, transaxial, axial,
                 fltnblut, scanner_name, description):
    # The number of gantries and layers is inferred from the size of the parameter lists
    n_gantries = len(radius)
    n_layers = [arr.size for arr in delta_rad]

    #
    n_crystals_per_module = (np.array(n_lat) * np.array(n_zed))[:, 0]
    target_shape = np.vstack((n_layers, n_rings, n_angles, n_crystals_per_module)).T
    # print(target_shape)

    # Allocate
    pos_x, pos_y, pos_z = [], [], []
    or_vx, or_vy, or_vz = [], [], []

    for ii in range(n_gantries):
        axial_rings_offset = symmetric_grid(n_rings[ii], dtype=fltnblut) * delta_rings[ii] + axial_offset[ii]
        phi = np.arange(n_angles[ii], dtype=fltnblut) / n_angles[ii] * 2 * np.pi
        # phi = np.arange(n_angles[ii], dtype=fltnblut) / n_angles[ii] * 2 * np.pi + (7.5 / 360 * 2 * np.pi)
        # print(phi / (2 * np.pi) * 360)

        # Following the CASToR documentation, this is first grouped in layers then rings
        for jj in range(n_layers[ii]):
            lat = symmetric_grid(n_lat[ii][jj], dtype=fltnblut) * delta_lat[ii][jj]
            zed = symmetric_grid(n_zed[ii][jj], dtype=fltnblut) * delta_zed[ii][jj]

            lat_layer, zed_layer = np.meshgrid(lat, zed, indexing='xy')
            lat_layer, zed_layer = lat_layer.flatten(order='C'), zed_layer.flatten(order='C')
            rad_layer = np.ones(lat_layer.shape, dtype=fltnblut) * radius[ii] + delta_rad[ii][jj]

            block_size = rad_layer.size

            for kk in range(n_rings[ii]):
                rad_ring = rad_layer.copy()
                lat_ring = lat_layer.copy()
                zed_ring = zed_layer + axial_rings_offset[kk]

                # Loop over the angles; the elements within one detector block are structured with the meshgrid above
                for ll in range(n_angles[ii]):
                    pos_x.append(rad_ring * np.cos(phi[ll]) - lat_ring * np.sin(phi[ll]))
                    pos_y.append(rad_ring * np.sin(phi[ll]) + lat_ring * np.cos(phi[ll]))
                    pos_z.append(zed_ring)

                    or_vx.append(np.cos(phi[ll]) * np.ones(block_size, dtype=fltnblut))
                    or_vy.append(np.sin(phi[ll]) * np.ones(block_size, dtype=fltnblut))
                    or_vz.append(np.zeros(block_size, dtype=fltnblut))

    pos_x, pos_y, pos_z = np.concatenate(pos_x), np.concatenate(pos_y), np.concatenate(pos_z)
    or_vx, or_vy, or_vz = np.concatenate(or_vx), np.concatenate(or_vy), np.concatenate(or_vz)

    # Check the normalization (should be normalized to one)
    # print(or_vx ** 2 + or_vy ** 2 + or_vz ** 2)

    # Estimate the expected number of elements
    n_elements_per_layer = (np.array(n_rings, ndmin=2) * np.array(n_angles, ndmin=2)).T * (np.array(n_lat) * np.array(n_zed))
    n_elements_per_gantry = np.sum(n_elements_per_layer, axis=1)
    n_elements = np.sum(n_elements_per_gantry)

    if n_elements != pos_x.size:
        sys.exit('Error: something is wrong with the number of LUT entries.')

    # Allocate a numpy structured array
    d_type = np.dtype([('Posx', fltnblut), ('Posy', fltnblut), ('Posz', fltnblut),
                       ('OrVx', fltnblut), ('OrVy', fltnblut), ('OrVz', fltnblut)])
    lut = np.empty(n_elements, dtype=d_type)  # Use np.empty(n, dtype=dtype)

    # Write the data
    lut['Posx'], lut['Posy'], lut['Posz'] = pos_x, pos_y, pos_z
    lut['OrVx'], lut['OrVy'], lut['OrVz'] = or_vx, or_vy, or_vz

    # Visualization
    visualize_lut_entries(lut, n_elements_per_layer.flatten(order='C'))

    # Save as binary file
    open(scanner_name + '.lut', 'wb').write(lut.tobytes())

    # Write the header file
    generate_hscan_file(scanner_name, description, lut.size, n_layers, n_elements_per_layer.flatten(order='C'),
                        np.array(depth).flatten(order='C'), np.array(transaxial).flatten(order='C'), np.array(axial).flatten(order='C'))

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
        for jj in range(0, z_slices.size, 10):
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
