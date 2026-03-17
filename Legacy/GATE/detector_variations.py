"""
Visualize Geometrical variation of the brain insert PET detector

@author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary functions
from Other.Gate_9_0.analyze_secondaries import add_boxes


def main():

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(projection='3d')
    # ax = fig.add_subplot()
    # ax.set_aspect('equal')


    # 6 x 6 SiPMs
    width = 6.0  # mm
    gap = 0.6  # mm
    n = 16
    tot = n * width + (n-1) * gap
    empty_fraction = 100 * (n-1) * gap / tot
    print('%d mm: %1.2f%%' % (width, empty_fraction))
    add_boxes(ax, [width, 30], [gap, 0], [n, 1], 16., 0., edge_color='black')

    # 4 x 4 SiPMs
    width = 4.0  # mm
    gap = 13/22  # mm
    n = 23
    tot = n * width + (n-1) * gap
    empty_fraction = 100 * (n-1) * gap / tot
    print('%d mm: %1.2f%%' % (width, empty_fraction))
    add_boxes(ax, [width, 18], [gap, 0], [n, 1], -10., 0., edge_color='tab:red')
    # add_boxes(ax, [4, 30], [17/21, 0], [22, 1], 0., 0., edge_color='tab:orange')

    # # 3 x 3 SiPMs
    # width = 3.0  # mm
    # gap = 9/14  # mm
    # n = 29
    # tot = n * width + (n-1) * gap
    # empty_fraction = 100 * (n-1) * gap / tot
    # print('%d mm: %1.2f%%' % (width, empty_fraction))
    # add_boxes(ax, [width, 30], [gap, 0], [n, 1], -31., 0., edge_color='tab:green')
    # # add_boxes(ax, [3, 30], [15/29, 0], [30, 1], -31., 0., edge_color='tab:green')

    ax.set_xlim(-60, 60)
    ax.set_ylim(-22, 35)
    ax.set_aspect(1)
    ax.set_xlabel(r'$x$ [mm]')
    ax.set_ylabel(r'$y$ [mm]')

    plt.show()

    return 0


def main2():

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    # 6 x 6 SiPMs
    width = 6.0  # mm
    gap = 0.6  # mm
    n = 16
    tot = n * width + (n-1) * gap
    empty_fraction = 100 * (n-1) * gap / tot
    print('%d mm: %1.2f%%' % (width, empty_fraction))
    add_boxes(ax, [width, 30], [gap, 0], [n, 1], 16.8, 0., edge_color='black')
    add_boxes(ax, [width, 30], [gap, 0], [n, 1], -16.8, 0., edge_color='black')

    # 6 x 6 SiPMs
    width = 108.15  # mm
    gap = 0.0  # mm
    n = 1
    tot = n * width + (n-1) * gap
    empty_fraction = 100 * (n-1) * gap / tot
    print('%d mm: %1.2f%%' % (width, empty_fraction))
    add_boxes(ax, [width, 3], [gap, 0], [n, 1], 0, 0., edge_color='black')

    ax.set_xlim(-60, 60)
    ax.set_ylim(-35, 35)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel(r'$x$ [mm]')
    # ax.set_ylabel(r'$y$ [mm]')

    plt.show()

    return 0


def main3():

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    # 4 x 4 SiPMs
    width = 4.0  # mm
    gap = 13/22  # mm
    n = 23
    tot = n * width + (n-1) * gap
    empty_fraction = 100 * (n-1) * gap / tot
    print('%d mm: %1.2f%%' % (width, empty_fraction))
    add_boxes(ax, [width, 18], [gap, 0], [n, 1], 10.8, 0., edge_color='black')
    add_boxes(ax, [width, 18], [gap, 0], [n, 1], -10.8, 0., edge_color='black')

    # 6 x 6 SiPMs
    width = 108.15  # mm
    gap = 0.0  # mm
    n = 1
    tot = n * width + (n-1) * gap
    empty_fraction = 100 * (n-1) * gap / tot
    print('%d mm: %1.2f%%' % (width, empty_fraction))
    add_boxes(ax, [width, 3], [gap, 0], [n, 1], 0, 0., edge_color='black')


    ax.set_xlim(-60, 60)
    ax.set_ylim(-35, 35)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    return 0


def main4():

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    # 4 x 4 SiPMs
    width = 4.0  # mm
    gap = 0.6  # mm
    n = 16
    tot = n * width + (n-1) * gap
    empty_fraction = 100 * (n-1) * gap / tot
    print('%d mm: %1.2f%%' % (width, empty_fraction))
    add_boxes(ax, [width, 18], [gap, 0], [n, 1], 11.3, 0., edge_color='black')
    add_boxes(ax, [width, 18], [gap, 0], [n, 1], -11.3, 0., edge_color='black')
    print(n * width + (n-1) * gap)

    # 6 x 6 SiPMs
    width = 75  # mm
    gap = 0.0  # mm
    n = 1
    tot = n * width + (n-1) * gap
    empty_fraction = 100 * (n-1) * gap / tot
    print('%d mm: %1.2f%%' % (width, empty_fraction))
    add_boxes(ax, [width, 4], [gap, 0], [n, 1], 0, 0., edge_color='black')


    ax.set_xlim(-40, 40)
    ax.set_ylim(-35, 35)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    return 0


def main5():

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    # 4 x 4 SiPMs
    width = 6.0  # mm
    # gap = 0.6  # mm
    gap = 0.7  # mm
    n = 11
    tot = n * width + (n-1) * gap
    empty_fraction = 100 * (n-1) * gap / tot
    print('%d mm: %1.2f%%' % (width, empty_fraction))
    add_boxes(ax, [width, 30], [gap, 0], [n, 1], 16.8, 0., edge_color='black')
    add_boxes(ax, [width, 30], [gap, 0], [n, 1], -16.8, 0., edge_color='black')
    print(n * width + (n-1) * gap)

    # 6 x 6 SiPMs
    width = 75  # mm
    gap = 0.0  # mm
    n = 1
    tot = n * width + (n-1) * gap
    empty_fraction = 100 * (n-1) * gap / tot
    print('%d mm: %1.2f%%' % (width, empty_fraction))
    add_boxes(ax, [width, 3], [gap, 0], [n, 1], 0, 0., edge_color='black')


    ax.set_xlim(-40, 40)
    ax.set_ylim(-35, 35)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    return 0


if __name__ == "__main__":
    # main()

    # main2()
    # main3()

    main4()
    main5()
