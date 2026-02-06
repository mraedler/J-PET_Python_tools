"""
Write volumetric data as interfile images

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Auxiliary functions
from read_interfile import read_interfile
# from vis import vis_3d


def main():
    input_dir = '/home/martin/PycharmProjects/J-PET/Sensitivity_maps/SiPM_6mm_depth_3cm'
    sensitivity_map = np.load(input_dir + '/accumulated.npy')
    x, y, z = np.load(input_dir + '/x.npy'), np.load(input_dir + '/y.npy'), np.load(input_dir + '/z.npy')
    # vis_3d(sensitivity_map, spacing=[10, 10, 20], transpose=True, axis=0)

    return 0


def write_sensitivity_map(x_edges, y_edges, z_edges, sensitivity_map, output_path, name, scanner):
    # From the grid vectors, extract spacing todo: and offset
    dx, dy, dz = x_edges[1] - x_edges[0], y_edges[1] - y_edges[0], z_edges[1] - z_edges[0]

    # Header
    dims = np.array(sensitivity_map.shape)
    voxel_sizes = np.array([dx, dy, dz])  # [mm]
    offset = np.array([np.mean(x_edges), np.mean(y_edges), np.mean(z_edges)])  # [mm]
    write_interfile_header(output_path, name, scanner, dims, voxel_sizes, offset)

    # Binary
    write_interfile_binary(sensitivity_map, output_path, name)
    return 0


def write_interfile_header(output_path, name, scanner, dims, voxel_sizes, offset):
    hdr_file = open(output_path + name + '.hdr', 'w')
    hdr_file.write('!INTERFILE :=\n')
    hdr_file.write('!imaging modality := PET\n')
    hdr_file.write('!version of keys := CASToRv3.1.1\n')
    hdr_file.write('CASToR version := 3.1.1\n')
    hdr_file.write('\n')

    hdr_file.write('!GENERAL DATA :=\n')
    hdr_file.write('!originating system := %s\n' % scanner)
    hdr_file.write('!data offset in bytes := 0\n')
    hdr_file.write('!name of data file := %s.img\n' % name)
    hdr_file.write('patient name := %s\n' % name)
    hdr_file.write('\n')

    hdr_file.write('!GENERAL IMAGE DATA\n')
    hdr_file.write('!type of data := Dynamic\n')
    hdr_file.write('!total number of images := %d\n' % 0)  # 2430
    hdr_file.write('imagedata byte order := LITTLEENDIAN\n')
    hdr_file.write('!number of frame groups := %d\n' % 1)
    hdr_file.write('process status := \n')
    hdr_file.write('\n')

    hdr_file.write('!STATIC STUDY (General) :=\n')
    hdr_file.write('number of dimensions := %d\n' % dims.size)
    for ii in range(dims.size):
        hdr_file.write('!matrix size [%d] := %d\n' % (ii + 1, dims[ii]))
    hdr_file.write('!number format := short float\n')
    hdr_file.write('!number of bytes per pixel := 4\n')
    for ii in range(voxel_sizes.size):
        hdr_file.write('scaling factor (mm/pixel) [%d] := %1.2f\n' % (ii + 1, voxel_sizes[ii]))
    for ii in range(dims.size):
        hdr_file.write('first pixel offset (mm) [%d] := %1.2f\n' % (ii + 1, offset[ii]))
    hdr_file.write('data rescale offset := 0\n')
    hdr_file.write('data rescale slope := 1\n')
    hdr_file.write('quantification units := 1\n')
    hdr_file.write('!number of images in this frame group := %d\n' % 0)  # 2430
    hdr_file.write('!image duration (sec) := 10\n')
    hdr_file.write('!image start time (sec) := 0\n')
    hdr_file.write('pause between frame groups (sec) := 0\n')
    hdr_file.write('!END OF INTERFILE :=\n')

    # !COPY OF INPUT HEADER 1
    # Scanner name: TB_JPET_6th_gen_7_rings_gap_2cm_Brain
    # Data filename: test.cdf
    # Number of events: 347419
    # Data mode: list-mode
    # Data type: PET
    # Start time (s): 0
    # Duration (s): 10
    # !END OF COPY OF INPUT HEADER 1

    hdr_file.close()
    return


def write_interfile_binary(volumetric_data, output_path, name):
    binary_file_path = output_path + name + '.img'
    binary_file = open(binary_file_path, 'wb')
    binary_file.write(volumetric_data.astype(np.float32).flatten(order='F').tobytes())
    binary_file.close()
    return 0


if __name__ == "__main__":
    main()
