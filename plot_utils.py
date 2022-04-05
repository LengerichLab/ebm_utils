from interpret.visual.plot import plot_bar
import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

def plot_all_bools(feat_names, ebm_global, mpl_style=False,
    figname=None, figsize=(12, 12), min_samples=None, ticksize=26):
    names = []
    upper_bounds = []
    impacts = []
    lower_bounds = []
    densities = []
    counter = 0
    for i, feat_name in enumerate(ebm_global.feature_names):
        if feat_name in feat_names:
            my_data = ebm_global.data(i)
            if len(my_data['scores']) == 2:
                my_name = "{} ({})".format(feat_name, my_data['density']['scores'][1])
                names.append(my_name)
                impacts.append(my_data['scores'][1]-my_data['scores'][0])
                upper_bounds.append(my_data['upper_bounds'][1] - my_data['lower_bounds'][0])
                lower_bounds.append(my_data['lower_bounds'][1] - my_data['upper_bounds'][0])
                densities.append(my_data['density']['scores'][1])
                counter += 1
            else:
                print("Feature: {} is not observed as a Boolean variable.".format(feat_name))
    if mpl_style:
        fig = plt.figure(figsize=figsize)
        sorted_i = np.argsort(impacts)
        for counter, i in enumerate(sorted_i):
            plt.bar(counter, impacts[i], width=0.5, color='blue', edgecolor='black',
                   yerr=upper_bounds[i]-impacts[i]) # Assume symmetric error.
        plt.xticks(range(len(names)), np.array(names)[sorted_i], rotation=90, fontsize=ticksize)
        plt.ylabel("Addition to Score", fontsize=32)
        plt.yticks(fontsize=ticksize)
        if figname is not None:
            plt.savefig(figname, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    else:
        sorted_i = np.argsort(impacts)
        names = np.array(names)[sorted_i]
        impacts = np.array(impacts)[sorted_i]
        upper_bounds = np.array(upper_bounds)[sorted_i]
        lower_bounds = np.array(lower_bounds)[sorted_i]
        densities_dict = {'names': names,
                          'scores': np.array(densities)[sorted_i]}
        data_dict = {'type': 'univariate',
            'names': names,
            'scores': impacts,
            'scores_range': (np.min(lower_bounds), np.max(upper_bounds)),
            'upper_bounds': upper_bounds,
            'lower_bounds': lower_bounds,
            'density': densities_dict,
            }
        return plot_bar(data_dict)