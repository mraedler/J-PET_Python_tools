"""
Average human measures to justify the chosen phantom dimensions

@author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt


def main():
    # ANSUR
    ansur_men = np.genfromtxt(sys.path[0] + '/Average_human_measures/ansurMen.csv', delimiter=',', names=True)
    ansur_women = np.genfromtxt(sys.path[0] + '/Average_human_measures/ansurWomen.csv', delimiter=',', names=True)

    for name in ansur_men.dtype.names:
        if name.startswith('HEAD'):
            print(name)
        if name.startswith('SHOULDER'):
            print(name)
        if name.startswith('HIP'):
            print(name)

    # plot_histograms(ansur_men['HEAD_LNTH'], ansur_women['HEAD_LNTH'], 165, 220, 2, 'Head length [mm]')
    # plot_histograms(ansur_men['HEAD_BRTH'], ansur_women['HEAD_BRTH'], 128, 170, 2, 'Head breadth [mm]')

    # ANSUR II
    ansur_ii_male = read_csv(sys.path[0] + '/Average_human_measures/ANSUR_II_MALE_Public.csv', encoding='ISO-8859-1')
    ansur_ii_female = read_csv(sys.path[0] + '/Average_human_measures/ANSUR_II_FEMALE_Public.csv', encoding='ISO-8859-1')

    print()
    # for column in ansur_ii_male.columns.tolist():
    #     # if column.startswith('head'):
    #     #     print(column)
    #     # if column.startswith('shoulder'):
    #     #     print(column)
    #     # if column.startswith('hip'):
    #     #     print(column)
    #     # if column.startswith('bidel'):
    #     #     print(column)
    #     if column.startswith('stat'):
    #         print(column)

    # plot_histograms(ansur_ii_male['headlength'].to_numpy(), ansur_ii_female['headlength'].to_numpy(), 165, 235, 2, 'Head length [mm]')
    # plot_histograms(ansur_ii_male['headbreadth'].to_numpy(), ansur_ii_female['headbreadth'].to_numpy(), 128, 180, 2, 'Head breadth [mm]')
    # plot_histograms(ansur_ii_male['hipbreadth'].to_numpy(), ansur_ii_female['hipbreadth'].to_numpy(), 270, 450, 4, 'Hip breadth [mm]')
    # plot_histograms(ansur_ii_male['hipbreadthsitting'].to_numpy(), ansur_ii_female['hipbreadthsitting'].to_numpy(), 290, 510, 4, 'Hip breadth sitting [mm]')
    # plot_histograms(ansur_ii_male['bideltoidbreadth'].to_numpy(), ansur_ii_female['bideltoidbreadth'].to_numpy(), 350, 650, 4, 'Bideltoid breadth [mm]')
    # plot_histograms(ansur_ii_male['bideltoidbreadth'].to_numpy(), ansur_ii_female['bideltoidbreadth'].to_numpy(), 350, 650, 4, 'Bideltoid breadth [mm]')
    # plot_histograms(ansur_ii_male['cervicaleheight'].to_numpy(), ansur_ii_female['cervicaleheight'].to_numpy(), 1100, 1800, 8, 'Cervicale height [mm]')
    plot_histograms(ansur_ii_male['acromialheight'].to_numpy(), ansur_ii_female['acromialheight'].to_numpy(), 1150, 1700, 8, 'Acromial height [mm]')


    plot_histograms(ansur_ii_male['stature'].to_numpy() - ansur_ii_male['acromialheight'].to_numpy(),
                    ansur_ii_female['stature'].to_numpy() - ansur_ii_female['acromialheight'].to_numpy(),
                    230, 400, 4, 'Stature - acromial height [mm]')

    return 0


def plot_histograms(data_male, data_female, bin_min, bin_max, bin_step, label):
    bin_edges = np.arange(bin_min, bin_max + 1, bin_step) - 1 / 2
    hist_male, _ = np.histogram(data_male, bins=bin_edges)
    hist_female, _ = np.histogram(data_female, bins=bin_edges)

    hist_male = hist_male / np.sum(hist_male) * 100
    hist_female = hist_female / np.sum(hist_female) * 100
    median_male = np.median(data_male)
    median_female = np.median(data_female)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax.stairs(hist_male, bin_edges, color='tab:blue', label='Male\n(n = %d)' % data_male.size)
    ax.stairs(hist_female, bin_edges, color='tab:orange', label='Female\n(n = %d)' % data_female.size)
    y_lim = ax.get_ylim()
    ax.plot([median_male, median_male], y_lim, color='tab:blue', linestyle='--')
    ax.plot([median_female, median_female], y_lim, color='tab:orange', linestyle='--')
    ax.set_ylim(y_lim)

    ax_top = ax.secondary_xaxis('top')
    ax_top.set_xticks([median_female, median_male])
    # ax_top.set_xticklabels(['median female = %d' % median_female, '%d = median male' % median_male])
    top_labels = ax_top.get_xticklabels()
    top_labels[0].set_color('tab:orange')
    top_labels[1].set_color('tab:blue')
    # top_labels[0].set_ha('left')
    # top_labels[1].set_ha('right')

    ax.set_xlabel(label)
    ax.set_ylabel('Relative frequency [%]')
    ax.legend(loc='upper right', frameon=False)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
