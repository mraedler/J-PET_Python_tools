"""
Write the GATE instructions for the Derenzo phantom

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np

# Auxiliary functions
from get_derenzo_parameters import get_derenzo_parameters


def main():
    # get_triangles_parameters(visualize=True)
    x, y, r, activities = get_derenzo_parameters(scaling_factor=3., visualize=False)

    # Additional parameters
    # z_shift = -815.0  # mm (previous version)
    # z_shift = 755.0  # mm (inside the brain insert)
    z_shift = -755.0  # mm (on the opposite side of the brian insert)

    insert_length = 50.0  # mm
    phantom_length = insert_length + 20.  # mm
    phantom_radius = 70.  # mm

    colors = np.array(['white', 'green', 'blue', 'cyan', 'magenta', 'yellow'])

    # mac_file = open('/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/Phantoms/Derenzo_Cox_3.mac', 'w')
    mac_file = open('/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/Phantoms/Derenzo_Cox_3_outside.mac', 'w')

    mac_file.write('#=====================================================\n'
                   '#   PYTHON GENERATED GATE CODE FOR THE CONSTRUCTION\n'
                   '#   OF THE DERENZO PHANTOM PUBLISHED IN\n'
                   '#   Am J Nucl Med Mol Imaging 2016;6(3):199-204\n'
                   '#=====================================================\n')

    mac_file.write('\n# PMMA container\n')
    add_phantom_cylinder(mac_file, 'world', 'derenzo', [0, 0, z_shift],
                         phantom_radius, phantom_length, 'PMMA', 'white')

    r_max = 0

    for ii in range(len(x)):
        mac_file.write('\n# Segment %d\n###########\n' % ii)
        for jj in range(x[ii].size):
            mac_file.write('\n# Rod %d\n' % jj)
            source_name = 'insert_%d_%d' % (ii, jj)

            r_center = np.sqrt(x[ii][jj] ** 2 + y[ii][jj] ** 2)
            r_outer = r_center + r[ii][jj]

            if r_outer > r_max:
                r_max = r_outer

            # Z-shift set to zero, since it is relative to the mother volume
            add_phantom_cylinder(mac_file, 'derenzo', source_name,
                                 [x[ii][jj], y[ii][jj], 0], r[ii][jj], insert_length, 'Water', 'blue')

    print(r_max)

    mac_file.close()

    return 0


def add_phantom_cylinder(mac_file, mother_volume_name, daughter_volume_name, center, radius, height, material, color):
    mac_file.write('/gate/%s/daughters/name %s\n' % (mother_volume_name, daughter_volume_name))
    mac_file.write('/gate/%s/daughters/insert cylinder\n' % mother_volume_name)

    mac_file.write('/gate/%s/placement/setTranslation %1.3f %1.3f %1.3f mm\n' % (daughter_volume_name, center[0], center[1], center[2]))
    mac_file.write('/gate/%s/geometry/setRmin 0. mm\n' % daughter_volume_name)
    mac_file.write('/gate/%s/geometry/setRmax %1.3f mm\n' % (daughter_volume_name, radius))
    mac_file.write('/gate/%s/geometry/setHeight %1.3f mm\n' % (daughter_volume_name, height))
    mac_file.write('/gate/%s/setMaterial %s\n' % (daughter_volume_name, material))

    mac_file.write('/gate/%s/vis/forceWireframe\n' % daughter_volume_name)
    mac_file.write('/gate/%s/vis/setColor %s\n' % (daughter_volume_name, color))

    mac_file.write('/gate/%s/attachPhantomSD\n' % daughter_volume_name)
    return 0


if __name__ == "__main__":
    main()
