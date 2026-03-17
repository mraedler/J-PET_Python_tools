"""
Edit the sensitivity map of a CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
from tqdm import trange

# Auxiliary functions
from read_interfile import read_interfile


def main():
    old_hdr_dir = '/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/derenzo/bb'
    old_hdr_path_template = old_hdr_dir + '/img_TB_brain_bb_all_GATE_it%d.hdr'

    new_dir = '/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/derenzo/bb22'
    new_name_template = 'img_TB_brain_bb_all_CONST_it%d'

    for ii in trange(1, 1000 + 1):
        # Original header
        old_hdr_path = old_hdr_path_template % ii
        new_hdr_path = new_dir + '/' + new_name_template % ii + '.hdr'

        # Load the associated volume
        vol_old = read_interfile(old_hdr_path)

        # Create a new volume, if desired
        # new_vol = np.ones(vol_old.shape, dtype=vol_old.dtype)
        new_vol = vol_old.copy()

        #
        new_vol_name = new_name_template % ii + '.img'

        # Copy the new header and exchange the name
        write_new_header(old_hdr_path, new_hdr_path, new_vol_name)

        # Save the new volume
        binary_file = open(new_dir + '/' + new_vol_name, 'wb')
        binary_file.write(new_vol.flatten(order='F').tobytes())
        binary_file.close()

    return 0


def write_new_header(old_hdr_path, new_hdr_path, new_vol_name):
    # Tag to be changed
    tag = '!name of data file := '

    old_hdr = open(old_hdr_path, 'r')
    new_hdr = open(new_hdr_path, 'w')

    for line in old_hdr:
        if line.startswith(tag):
            line = tag + new_vol_name + '\n'
        new_hdr.write(line)

    old_hdr.close()
    new_hdr.close()
    return 0


if __name__ == "__main__":
    main()
