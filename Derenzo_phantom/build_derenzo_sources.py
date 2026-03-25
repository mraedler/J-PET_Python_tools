"""
Write the GATE instructions for the sources in the Derenzo phantom

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np

# Auxiliary functions
from get_derenzo_parameters import get_derenzo_parameters


def main():
    # get_triangles_parameters(visualize=True)
    x, y, r, activities = get_derenzo_parameters(scaling_factor=3., visualize=True)
    sys.exit()

    # Additional parameters
    # z_shift = -815.0  # mm (previous version)
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
            # add_cylinder_gate(mac_file, ii, jj, x[ii][jj], y[ii][jj], z_shift, r[ii][jj], length, activities[ii], colors[ii], add_non_collinearity)
            add_cylinder_gate_v2(mac_file, ii, jj, x[ii][jj], y[ii][jj], z_shift, r[ii][jj], length, activities[ii], colors[ii], add_non_collinearity)

    mac_file.close()

    return 0


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


def add_cylinder_gate_v2(mac_file, sec_idx, rod_idx, x, y, z, r, h, activity, color, add_non_collinearity):
    source_name = 'rod_%d_%d' % (sec_idx, rod_idx)
    mac_file.write('/gate/source/addSource %s\n' % source_name)
    mac_file.write('/gate/source/%s/setType backtoback\n' % source_name)
    if add_non_collinearity:
        mac_file.write('/gate/source/%s/setAccolinearityFlag True\n' % source_name)
        mac_file.write('/gate/source/%s/setAccoValue 0.5 deg\n' % source_name)
    mac_file.write('/gate/source/%s/gps/pos/type Volume\n' % source_name)
    mac_file.write('/gate/source/%s/gps/pos/shape Cylinder\n' % source_name)
    mac_file.write('/gate/source/%s/gps/pos/radius %1.3f mm\n' % (source_name, r))
    mac_file.write('/gate/source/%s/gps/pos/halfz %1.3f mm\n' % (source_name, h / 2))
    mac_file.write('/gate/source/%s/gps/pos/centre %1.3f %1.3f %1.3f mm\n' % (source_name, x, y, z))
    mac_file.write('/gate/source/%s/gps/particle gamma\n' % source_name)
    mac_file.write('/gate/source/%s/gps/ene/type Mono\n' % source_name)
    mac_file.write('/gate/source/%s/gps/ene/mono 511 keV\n' % source_name)
    mac_file.write('/gate/source/%s/gps/ang/type iso\n' % source_name)
    mac_file.write('/gate/source/%s/setActivity %1.3f Bq\n' % (source_name, activity))
    mac_file.write('/gate/source/%s/visualize 100 %s 5\n' % (source_name, color))
    return


if __name__ == "__main__":
    main()
