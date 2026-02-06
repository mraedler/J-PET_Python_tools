"""
Read the interfile format of CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
from os.path import basename, dirname
from glob import glob
from natsort import natsorted
import numpy as np
from re import split


def main():
    # Example input
    hdr_path = '/home/martin/J-PET/CASToR_scripts/recon/sensitivity_maps/TB_only_sensitivity.hdr'
    vol = read_interfile(hdr_path)
    print(vol.shape)
    return 0


def read_interfile(hdr_path, return_grid=False):
    header_data = read_interfile_header(hdr_path)

    # Construct the path of the binary file
    img_path = dirname(hdr_path) + '/' + header_data['!name of data file']

    # Load the volume todo: Generalize the data type
    vol = np.fromfile(img_path, dtype=np.float32, offset=int(header_data['!data offset in bytes']))

    # Get the volume shape todo: Generalize to n dimensions
    vol_shape = (int(header_data['!matrix size [1]']),
                 int(header_data['!matrix size [2]']),
                 int(header_data['!matrix size [3]']))

    # Reshape
    if header_data['imagedata byte order'] == 'LITTLEENDIAN':
        vol = np.reshape(vol, vol_shape, order='F')
    elif header_data['imagedata byte order'] == 'BIGENDIAN':
        vol = np.reshape(vol, vol_shape, order='C')
    else:
        sys.exit('Error: "imagedata byte order" not known.')

    if return_grid:
        # Estimate the grid vectors from the header data
        n_dim = int(header_data['number of dimensions'])
        v = []
        for ii in range(1, n_dim + 1):
            n = int(header_data['!matrix size [%d]' % ii])
            spacing = float(header_data['scaling factor (mm/pixel) [%d]' % ii])
            offset = float(header_data['first pixel offset (mm) [%d]' % ii])
            v.append(symmetric_grid(n) * spacing + offset)

        return *v, vol
    else:
        return vol


def read_interfile_header(hdr_path):
    # Get the number of dimensions
    general_tags = ['!name of data file', '!data offset in bytes', '!number of bytes per pixel', 'imagedata byte order', 'number of dimensions']
    general_data = search_ascii_file(hdr_path, general_tags, ' :=')
    n_dim = int(general_data[-1])

    # Construct the remaining tags based on the number of dimensions
    volume_tags = [''] * n_dim * 3
    matrix_size = '!matrix size [%d]'
    scaling_factor = 'scaling factor (mm/pixel) [%d]'
    first_pixel_offset = 'first pixel offset (mm) [%d]'
    for ii in range(n_dim):
        volume_tags[ii + 0 * n_dim] = matrix_size % (ii + 1)
        volume_tags[ii + 1 * n_dim] = scaling_factor % (ii + 1)
        volume_tags[ii + 2 * n_dim] = first_pixel_offset % (ii + 1)
    # print(volume_tags)

    # Read the data
    volume_data = search_ascii_file(hdr_path, volume_tags, ' :=')

    # Convert to dict
    header_data = dict(zip(general_tags + volume_tags, general_data + volume_data))

    return header_data


def search_ascii_file(ascii_file_path, tags_list, tag_value_separator, comment_separator=None, n_entries=1, entries_separator=None):
    pattern = ' |\n'
    if comment_separator is not None:
        pattern += '|' + comment_separator

    if n_entries > 1:
        if entries_separator is not None:
            pattern += '|' + entries_separator
        else:
            sys.exit('Error: must provide an "entries_separator" for "n_entries" > 1.')

    ascii_file = open(ascii_file_path, 'r')
    out_list = []
    order_list = []
    for line in ascii_file:
        for ii in range(len(tags_list)):
            if line.startswith(tags_list[ii] + tag_value_separator):
                # Remove the tag
                line_without_tag = line[len(tags_list[ii]) + len(tag_value_separator):]

                # Split and remove empty spaces
                line_content = split(pattern, line_without_tag)
                line_content = remove_empty_characters(line_content)

                # Export
                if n_entries > 1:
                    out_list.append(line_content[:n_entries])
                else:
                    out_list.append(line_content[0])
                order_list.append(ii)

    out_list = [out_list[i] for i in np.argsort(np.array(order_list))]

    if len(out_list) != len(tags_list):
        print('Warning: Not all tags were found.')
    return out_list


def remove_empty_characters(input_list):
    output_list = []
    for element in input_list:
        if len(element) > 0:
            output_list.append(element)
    return output_list


def symmetric_grid(n):
    return np.arange(n) - (n - 1) / 2


def accumulate_slices(img_paths, idx_0=0, idx_1=None, slice_pos=0.5):
    # Get all image paths in order
    hdr_paths = natsorted(glob(img_paths))[idx_0:idx_1]

    # Load the first one to get the image grid
    x, y, z, _ = read_interfile(hdr_paths[0], return_grid=True)

    # Get the slice location
    if isinstance(slice_pos, float) and (slice_pos >= 0.) and (slice_pos < 1.):
        slice_idx = int(z.size * slice_pos)
    elif isinstance(slice_pos, int) and (slice_pos >= 0) and (slice_pos < z.size):
        slice_idx = slice_pos
    else:
        sys.exit('Error: slice_pos must be either a float within [0, 1) or a positive integer smaller than z.size!')

    img_acc = np.zeros((x.size, y.size, len(hdr_paths)))

    for ii in range(len(hdr_paths)):
        img = read_interfile(hdr_paths[ii], return_grid=False)
        img_acc[:, :, ii] = img[:, :, slice_idx]

    return img_acc


if __name__ == "__main__":
    main()
