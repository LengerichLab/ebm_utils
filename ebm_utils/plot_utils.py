from interpret.visual.plot import plot_bar
import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

def axvlines(xs, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = (0.0, 0.15, np.nan) #plt.gca().get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims)[None, :], repeats=len(xs), axis=0).flatten()
    plot = plt.plot(x_points, y_points, color='red', alpha=0.4, scaley = False, **plot_kwargs)
    return plot

def fill_x(orig):
    new_x = [orig[0]]
    for i in range(1, len(orig)):
        new_x.append(orig[i-1] + (orig[i] - orig[i-1]) / 2)
        new_x.append(orig[i-1] + (orig[i] - orig[i-1]) / 2)
        new_x.append(orig[i])
    return np.array(new_x)

def fill_y(orig):
    new_y = [orig[0]]
    for i in range(1, len(orig)):
        new_y.append(orig[i-1])
        new_y.append(orig[i])
        new_y.append(orig[i])
    return np.array(new_y)

def axline(x):
    plt.axvline(x, linestyle='--', color='gray')

def standardize(feat):
    return feat


def plot_feat(exp, feat, noise_levels = {}, axlines = {},
    ylims = {}, xlims = {}, xlabels = {}):
    i = exp.feature_names.index(feat)
    data = exp.data(i)
    if type(data['names'][0]) is str:
        return
    fig = plt.figure(figsize=(15, 10))
    my_min = np.min(data['scores'])

    standard_feat = standardize(feat.lower().strip())
    print(standard_feat)
    xlabel = xlabels.get(standard_feat, feat)
    for x in axlines.get(standard_feat, []):
        axline(x)
    my_xlims = xlims.get(standard_feat, [np.percentile(data['names'], 2), np.percentile(data['names'], 98)])
    my_ylims = ylims.get(standard_feat, [-0.0, 5.0])
    noise_level = noise_levels.get(standard_feat, 0.01)
    plt.xlabel(xlabel, fontsize=54)
    plt.xlim(my_xlims)
    plt.ylim(my_ylims)

    my_names = data['names']
    my_xs = X[feat]
    if 'tempc' in feat.lower():
        my_xs = my_xs * (9/5) + 32
        my_names = np.array(my_names)*(9/5) + 32
    if noise_level > 0:
        axvlines(sorted(my_xs+np.random.uniform(-noise_level, noise_level, size=my_xs.shape))[::50])
    my_min = np.min([data['scores'][j] for j, x in enumerate(my_names)
                     if x <= plt.xlim()[1] and x >= plt.xlim()[0]])
    lowers = np.exp(fill_y(data['lower_bounds']-my_min))
    uppers = np.exp(fill_y(data['upper_bounds']-my_min))
    means  = np.exp(fill_y(data['scores']-my_min))

    plt.fill_between(fill_x(my_names), means-2*(means-lowers),
                     means+2*(uppers-means), alpha=0.2)
    plt.plot(fill_x(my_names), means)
    plt.ylabel("Mortality Odds Ratio", fontsize=46)
    #plt.show()

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