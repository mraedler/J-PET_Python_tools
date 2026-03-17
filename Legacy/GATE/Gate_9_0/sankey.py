"""
Sankey diagram for the GATE processes

Author: Martin Rädler
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey


def main():
    # plt.rcParams.update({'font.size': 16})
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Two Systems")
    # flows = [0.25, 0.15, 0.60, -0.10, -0.05, -0.25, -0.15, -0.10, -0.35]
    # sankey = Sankey(ax=ax, unit=None)
    # sankey.add(flows=flows, label='one',
    #            orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0])
    # sankey.add(flows=[-0.25, 0.15, 0.1], label='two',
    #            orientations=[-1, -1, -1], prior=0, connect=(0, 0))
    # diagrams = sankey.finish()
    # diagrams[-1].patch.set_hatch('/')
    # plt.legend()
    # plt.show()

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="GATE physics (Hits)")
    # sankey = Sankey(ax=ax, scale=0.00000005, offset=0.2, head_angle=135, format='%1.1e', unit='')
    sankey = Sankey(ax=ax, scale=0.00000005, offset=0.2, head_angle=135, format='', unit='')

    n_0 = 20008648

    sankey.add(flows=[-8132155, -22357, -(20008648-8132155-22357), 20008648],
               labels=['Interact\n(%1.1f%%)' % (100 * 8132155 / n_0),
                       'Zero\ninteractions\n(%1.1f%%)' % (100 * 22357 / n_0),
                       'Zero\ninteractions\n(%1.1f%%)' % (100 * (20008648 - 8132155 - 22357) / n_0),
                       '511 keV\nphotons\n(100%)'],
               orientations=[0, 0, -1, 0],
               pathlengths=[0.25, 0.25, 0.25, 0.25],
               patchlabel='Geometry &\ninteraction\nprobability',
               trunklength=1.2)

    sankey.add(flows=[8132155, -7826942, -173133, -100816, - (8132155 - 8100891)],
               orientations=[0, -1, -1, -1, 0],
               labels=[None,
                       'Only\nCompton\n(%1.1f%%)' % (100 * 7826942 / n_0),
                       'Rayleigh &\n(Compton)\n(%1.1f%%)' % (100 * 173133 / n_0),
                       'Finish with\nPhotoelectric\n(%1.1f%%)' % (100 * 100816 / n_0),
                       'Creating\nsecondaries\n(%1.1f%%)' % (100 * (8132155 - 8100891) / n_0)],
               pathlengths=[0.25, 0.25, 0.75, 1.25, -0.3],
               trunklength=0.5,
               prior=0,
               connect=(0, 0))

    factor = 302.9
    n_1 = 22357 + 31264

    sankey.add(flows=[22357, -factor * 22357],
               labels=[None,
                       'W/o parent\n(%1.1f%%)' % (100 * 22357 / n_1)],
               orientations=[0, 0],
               pathlengths=[1.368, 0.25],
               prior=0,
               connect=(1, 0))

    sankey.add(flows=[31264, -factor * 588, -factor * 678, -factor * (31306 - 588 - 678)],
               orientations=[0, 0, 0, 0],
               labels=[None,
                       'Via Compton (%1.1f%%)' % (100 * 588 / n_1),
                       'Via photoelectric (%1.1f%%)' % (100 * 678 / n_1),
                       'W/ parent\n(%1.1f%%)' % (100 * (31306 - 588 - 678) / n_1)],
               pathlengths=[0.25, 0.25, 0.25, 0.25],
               trunklength=1,
               prior=1,
               connect=(4, 0))

    sankey.add(flows=[factor * 588, factor * 678, factor * (31306 - 588 - 678), factor * 22357, -factor * 1091, -factor * 742, -factor * (n_1 - 1091 - 742)],
               labels=[None,
                       None,
                       None,
                       None,
                       'Tertiary\nelectrons\n(%1.1f%%)' % (100 * 1091 / n_1),
                       'Tertiary\nphotons\n(%1.1f%%)' % (100 * 742 / n_1),
                       'W/o\ntertiaries\n(%1.1f%%)' % (100 * (n_1 - 1091 - 742) / n_1)],
               orientations=[0, 0, 0, 0, -1, -1, 0],
               pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
               prior=3,
               connect=(1, 0),
               trunklength=0.35,
               patchlabel='Electrons')

    diagrams = sankey.finish()

    # Adjust the text positions
    diagrams[0].text.set_weight('bold')
    diagrams[0].texts[0].set_position(xy=(1.25, 0.23))
    diagrams[0].texts[1].set_position(xy=(1.25, -0.12))
    diagrams[0].texts[3].set_position(xy=(-0.645, -0.07))

    diagrams[1].texts[4].set_position(xy=(2.4, 0.62))

    diagrams[2].texts[1].set_position(xy=(3.35, 0.02))

    diagrams[3].texts[1].set_position(xy=(3.35, 0.87))
    diagrams[3].texts[2].set_position(xy=(3.35, 0.75))
    diagrams[3].texts[3].set_position(xy=(3.35, 0.44))

    diagrams[4].text.set_weight('bold')
    diagrams[4].text.set_position(xy=(4.1, 0.35))

    diagrams[4].texts[4].set_position(xy=(4.30, -0.53))
    diagrams[4].texts[5].set_position(xy=(3.85, -0.53))

    plt.show()
    return 0


if __name__ == '__main__':
    main()