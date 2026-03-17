"""
Reading/writing binary LUT files from CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt


def main():
    # Reference data from Jakub Baran
    # lut = read_txt_lut('/home/martin/J-PET/CASToR_scripts/new/TB_JPET_6th_gen_7_rings_gap_2cm_original.txt')

    # Using castor-scannerLUTExplorer provided the adjusted .geom file: Brain scanner with 6 mm SiPMs
    # lut = read_txt_lut('/home/martin/J-PET/CASToR/castor/config/scanner/user_generated/TB_JPET_6th_gen_7_rings_gap_2cm_Brain.txt')
    # lut = read_txt_lut('/home/martin/J-PET/CASToR/castor/config/scanner/user_generated/TB_JPET_6th_gen_7_rings_gap_2cm_axial_6mm_Brain.txt')
    # lut = read_txt_lut('/home/martin/J-PET/CASToR/castor/config/scanner/user_generated/TB_JPET_6th_gen_7_rings_gap_2cm_axial_10mm_Brain.txt')
    # Data from the .geom file (The orientation vectors sit in the middle of the crystals)
    scanner_radii = np.array([414.799, 448.399, 445.099, 168.2, 201.8, 198.5], dtype=np.float32)
    crystal_depths = np.array([30, 30, 3, 30, 30, 3], dtype=np.float32)
    n_rsectors = np.array([24, 24, 24, 10, 10, 10], dtype=np.int32)
    z_shifts = np.array([0., 0., 0., -815., -815., -815.], dtype=np.float32)
    # group_lut_data(lut, scanner_radii, crystal_depths, z_shifts, n_rsectors)
    # sys.exit()

    # Using castor-scannerLUTExplorer provided the adjusted .geom file: Brain scanner with 4 mm SiPMs
    # lut = read_txt_lut('/home/martin/J-PET/CASToR/castor/config/scanner/user_generated/TB_JPET_6th_gen_7_rings_gap_2cm_Brain_2.txt')
    lut = read_lut_txt('/home/martin/J-PET/CASToR/castor/config/scanner/user_generated/TB_JPET_axial_3mm_B2_axial_3mm.txt')
    scanner_radii = np.array([414.799, 448.399, 445.099, 167.2, 189.8, 186.5], dtype=np.float32)
    crystal_depths = np.array([30, 30, 3, 18, 18, 3], dtype=np.float32)
    z_shifts = np.array([0., 0., 0., -815., -815., -815.], dtype=np.float32)
    n_rsectors = np.array([24, 24, 24, 10, 10, 10], dtype=np.int32)
    # group_lut_data(lut, scanner_radii, crystal_depths, z_shifts, n_rsectors)
    # sys.exit()

    #
    # lut = read_binary_lut('/home/martin/J-PET/CASToR/castor/config/scanner/TB_JPET_6th_gen_7_rings_gap_2cm_Brain.lut')
    # lut = read_binary_lut('/home/martin/J-PET/CASToR/castor/config/scanner/TB_JPET_6th_gen_7_rings_gap_2cm_Brain_2.lut')
    # lut = read_binary_lut('/home/martin/J-PET/CASToR/castor/config/scanner/TB_JPET_6th_gen_7_rings_gap_2cm_axial_6mm_Brain.lut')
    # lut = read_binary_lut('/home/martin/J-PET/CASToR/castor/config/scanner/TB_JPET_6th_gen_7_rings_gap_2cm_axial_10mm_Brain.lut')
    lut = read_lut_binary('TB_J-PET_7th_gen.lut')

    visualize_lut_entries(lut)

    return 0


def read_lut_binary(binary_file_path):
    d_type = np.dtype([('Posx', np.float32), ('Posy', np.float32), ('Posz', np.float32),
                       ('OrVx', np.float32), ('OrVy', np.float32), ('OrVz', np.float32)])
    return np.fromfile(binary_file_path, dtype=d_type)


def read_lut_header(header_file_path):
    # todo: read more entries, if necessary
    search_entries = ('number of crystals in layer',
                      'crystals size depth',
                      'crystals size transaxial',
                      'crystals size axial')

    header = {}
    file = open(header_file_path, 'r')
    for line in file:
        if line.startswith(search_entries):
            key, value = line.split(':', 1)
            # Remove trailing comments
            value = value.split('#')[0]

            # Simplify
            value = value.strip()

            # To numpy array
            value = np.fromstring(value, sep=',', dtype=float)

            # Write into dict
            header[key] = value
    file.close()

    # Cast the number of elements back to int
    header[search_entries[0]] = header[search_entries[0]].astype(int)

    return header


def read_lut_txt(txt_file_name):
    txt_file = open(txt_file_name, 'r')

    #
    x_list, y_list, z_list = [], [], []
    vx_list, vy_list, vz_list = [], [], []

    for line in txt_file:
        if line.startswith('Scanner element center location (x,y,z):'):
            line_split = line.split()

            x = np.float32(line_split[5])
            y = np.float32(line_split[7])
            z = np.float32(line_split[9])

            vx = np.float32(line_split[12])
            vy = np.float32(line_split[14])
            vz = np.float32(line_split[16])

            # print(line_split)
            # print(x, y, z, vx, vy, vz)

            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

            vx_list.append(vx)
            vy_list.append(vy)
            vz_list.append(vz)

    # Cast to structured array
    d_type = np.dtype([('Posx', np.float32), ('Posy', np.float32), ('Posz', np.float32),
                       ('OrVx', np.float32), ('OrVy', np.float32), ('OrVz', np.float32)])
    # Allocate
    lut = np.empty(len(x_list), dtype=d_type)

    # Probably does not need to be cast to np.array
    lut['Posx'] = np.array(x_list)
    lut['Posy'] = np.array(y_list)
    lut['Posz'] = np.array(z_list)
    lut['OrVx'] = np.array(vx_list)
    lut['OrVy'] = np.array(vy_list)
    lut['OrVz'] = np.array(vz_list)

    return lut


def write_lut_from_structured_array(binary_file_path, structured_array):
    binary_file = open(binary_file_path, 'wb')
    binary_file.write(structured_array.tobytes())
    binary_file.close()
    return 0


def group_lut_data(lut, scanner_radii, crystal_depths, z_shifts, n_rsectors):
    # Radius
    rho_lut = np.sqrt(lut['Posx'] ** 2 + lut['Posy'] ** 2)

    # The orientation vectors sit in the middle of the crystals
    rho = scanner_radii + crystal_depths / 2

    # Remove the WLSes
    no_wls = np.array([True, True, False, True, True, False], dtype=bool)
    rho, z_shifts, n_rsectors = rho[no_wls], z_shifts[no_wls], n_rsectors[no_wls]

    #
    # delta_rho = 7.5
    delta_rho = 8.5
    check_layer_separation(rho_lut, rho, delta_rho)

    # Separate into rings based on z
    z = (np.arange(8) - 3.5) * 350
    # z = (np.arange(8) - 3.5) * 339.6  # WLS
    # check_ring_separation(lut['Posz'], z)
    # sys.exit()

    # Separate into rsectors based on phi
    phi_24 = np.mod(np.arange(0, 360, 15) + 0, 360) / 360 * 2 * np.pi
    phi_10 = np.mod(np.arange(0, 360, 36) + 0, 360) / 360 * 2 * np.pi
    phi_dict = {10: phi_10, 24: phi_24}
    delta_phi = 1 / 360 * 2 * np.pi

    binary_file = open('test.lut', 'wb')

    # Separate into layers based on the radius
    for ii in range(rho.size):
        layer_temp = lut[(rho_lut < rho[ii] + delta_rho) & (rho_lut > rho[ii] - delta_rho)]
        print(layer_temp.size)

        # Separate into rings based on z
        z_lut = layer_temp['Posz']
        check_ring_separation(z_lut, z)

        for jj in range(z.size-1):
            ring_temp = layer_temp[(z_lut > z[jj]) & (z_lut < z[jj+1])]

            if ring_temp.size == 0:
                continue

            # Separate into r-sectors based on angles
            angles_lut = np.arctan2(ring_temp['OrVy'], ring_temp['OrVx']) + np.pi
            unique_angles = np.unique(angles_lut) / (2 * np.pi) * 360
            unique_angles = np.round(unique_angles).astype(int)

            if n_rsectors[ii] != unique_angles.size:
                sys.exit('Error')

            phi_temp = phi_dict[n_rsectors[ii]]

            for kk in range(0, phi_temp.size):
                rsector_temp = ring_temp[(angles_lut < phi_temp[kk] + delta_phi) & (angles_lut > phi_temp[kk] - delta_phi)]

                pos_x = rsector_temp['Posx']
                pos_y = -rsector_temp['Posy'] * -1
                pos_z = rsector_temp['Posz'] + z_shifts[ii]

                or_vx = rsector_temp['OrVx']
                or_vy = -rsector_temp['OrVy'] * -1
                or_vz = rsector_temp['OrVz']

                # Rotate back (should be unique afterward)
                or_vx_prime = or_vx * np.cos(-phi_temp[kk]) - or_vy * np.sin(-phi_temp[kk])
                or_vy_prime = or_vx * np.sin(-phi_temp[kk]) + or_vy * np.cos(-phi_temp[kk])
                pos_x_prime = pos_x * np.cos(-phi_temp[kk]) - pos_y * np.sin(-phi_temp[kk])
                pos_y_prime = pos_x * np.sin(-phi_temp[kk]) + pos_y * np.cos(-phi_temp[kk])
                print(np.unique(pos_x_prime))
                print(np.unique(pos_y_prime))

                for ll in range(pos_x.size):
                    binary_file.write(pos_x[ll])
                    binary_file.write(pos_y[ll])
                    binary_file.write(pos_z[ll])
                    binary_file.write(or_vx[ll])
                    binary_file.write(or_vy[ll])
                    binary_file.write(or_vz[ll])

    binary_file.close()
    return 0


def check_layer_separation(rho_lut, rho, delta_rho):
    # Check the radius separation
    rho_edges = np.linspace(0, 500, 1001)
    rho_centers = (rho_edges[1:] + rho_edges[:-1]) / 2
    rho_width = rho_edges[1:] - rho_edges[:-1]
    h, _ = np.histogram(rho_lut, bins=rho_edges)

    fig, ax = plt.subplots()
    ax.bar(rho_centers, h, width=rho_width)
    y_lim = ax.get_ylim()
    radius_mesh, y_lim_mesh = np.meshgrid(rho, y_lim)
    ax.plot(radius_mesh, y_lim_mesh, color='black')
    ax.bar(rho, y_lim[1], width=2*delta_rho, alpha=0.5)
    ax.set_ylim(y_lim)
    plt.show()
    return 0


def check_ring_separation(z_lut, z):
    # Check the separation of the rings
    z_edges = np.linspace(-1500, 1500, 3001)
    z_centers = (z_edges[1:] + z_edges[:-1]) / 2
    z_width = z_edges[1:] - z_edges[:-1]
    h, _ = np.histogram(z_lut, bins=z_edges)

    fig, ax = plt.subplots()
    ax.bar(z_centers, h, width=z_width)
    y_lim = ax.get_ylim()
    z_mesh, y_lim_mesh = np.meshgrid(z, y_lim)
    ax.plot(z_mesh, y_lim_mesh, color='black')
    ax.set_ylim(y_lim)
    plt.show()
    return 0


def visualize_lut_entries(lut):
    # Group the data into z-slices
    z_slices = np.unique(lut['Posz'])

    # print(z_slices)
    # sys.exit()

    pos_x, pos_y, orv_x, orv_y = [], [], [], []
    n_elem = np.zeros((z_slices.size,), dtype=int)

    for ii in range(z_slices.size):
        # Choose the data from the different slices
        logical_index = lut['Posz'] == z_slices[ii]
        n_elem[ii] = np.sum(logical_index)

        # Drop in list
        pos_x.append(lut['Posx'][logical_index])
        pos_y.append(lut['Posy'][logical_index])
        orv_x.append(lut['OrVx'][logical_index])
        orv_y.append(lut['OrVy'][logical_index])

    # Categorize by count
    c = np.unique(n_elem)
    # c = np.flip(c)

    plt.rcParams.update({'font.size': 16})
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 4), width_ratios=(1, 2.5), sharey=True)
    for ii in range(c.size):
        idx_c = np.argwhere(n_elem == c[ii]).flatten()
        # ax0.plot(z_slices[idx_c], z_slices[idx_c], 'x-')
        ax0.quiver(pos_x[idx_c[0]], pos_y[idx_c[0]], orv_x[idx_c[0]], orv_y[idx_c[0]], angles='xy', scale_units='xy', scale=1 / 20, color=colors[ii])

        for jj in range(0, idx_c.size, 10):
            ax1.quiver(np.ones(pos_y[idx_c[jj]].shape) * z_slices[idx_c[jj]], pos_y[idx_c[jj]],
                       np.zeros(pos_y[idx_c[jj]].shape), orv_y[idx_c[jj]], angles='xy', scale_units='xy', scale=1 / 20, color=colors[ii])

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

