import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


mpl.rc('lines', linewidth=2)
mpl.rcParams.update(
    {'font.size': 13, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
mpl.rcParams['xtick.major.pad'] = 2
mpl.rcParams['ytick.major.pad'] = 2


def fancy_plot(values_list, colors, labels, markers, file_name, text):
    v_min = min([min(val) for val in values_list])
    plt.plot(figsize=(5, 4))
    for i, values in enumerate(values_list):
        new_values = np.array(values) - v_min
        N = len(values)
        plt.plot(new_values, colors[i], label=labels[i],
                 marker=markers[i], markevery=N / 10)
        plt.xlabel(u' iterations ')
        plt.ylabel(text)
        plt.legend(loc=0)
        plt.yscale('log')

    plt.minorticks_off()
    plt.tight_layout()
    # fig.show()
    plt.savefig(file_name)
