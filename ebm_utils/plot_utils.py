from interpret.visual.plot import plot_bar
import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import init_notebook_mode
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
ebc = ExplainableBoostingClassifier
ebr = ExplainableBoostingRegressor

init_notebook_mode(connected=True)



def axvlines(xs, **kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = (0.0, plt.gca().get_ylim()[-1]*0.05, np.nan)
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims)[None, :], repeats=len(xs), axis=0).flatten()
    plot = plt.plot(x_points, y_points, color='red', alpha=0.4, scaley = False, **kwargs)
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
    # TODO
    return feat


def plot_discrete(data, xlims, my_names):
    my_xlims = [-0.5, len(my_names)-0.5]
    my_min = np.min([data['scores'][j] for j, x in enumerate(my_names)
                     if x <= plt.xlim()[1] and x >= plt.xlim()[0]])
    lowers = np.exp(fill_y(data['lower_bounds']-my_min))
    uppers = np.exp(fill_y(data['upper_bounds']-my_min))
    means  = np.exp(fill_y(data['scores']-my_min))

    if len(my_names) > 5:
        plt.xticks(rotation=60, ha='right')
    return my_names, my_xlims, lowers, uppers, means

def repeat_last(np_ar):
    return np.append(np_ar, np_ar[-1])


def plot_numeric(data, xlims, standard_feat):
    my_xlims = xlims.get(standard_feat, [np.percentile(data['names'], 2), np.percentile(data['names'], 98)])
    my_names = data['names']

    my_min = np.min([repeat_last(data['scores'])[j] for j, x in enumerate(my_names)
                     if x <= plt.xlim()[1] and x >= plt.xlim()[0]])
    lowers = np.exp(fill_y(repeat_last(data['lower_bounds']-my_min)))
    uppers = np.exp(fill_y(repeat_last(data['upper_bounds']-my_min)))
    means  = np.exp(fill_y(repeat_last(data['scores']-my_min)))
    return my_names, my_xlims, lowers, uppers, means

def ebm_marginalize(X, Y, feat_name, xlims={}, ylims={}, noise_levels={}, ylabel="Odds Ratio", plot_every=50,
    classification=True,
                    **kwargs):
    if classification:
        ebm = ebc(interactions=0, **kwargs)
    else:
        ebm = ebr(interactions=0, **kwargs)
    ebm.fit(X, Y)
    exp = ebm.explain_global()
    data = exp.data(0)
    fig = plt.figure(figsize=(15, 10))
    plt.xlabel(feat_name, fontsize=54)
    try:
        my_names, my_xlims, lowers, uppers, means = plot_numeric(data, xlims, feat_name)
        default_noise_level = 0.01
    except:
        my_names, my_xlims, lowers, uppers, means = plot_discrete(data, xlims, list(set(X[feat_name].values)))
        default_noise_level = 0.1
    plt.xlim(my_xlims)
    if feat_name in ylims:
        plt.ylim(ylims[feat_name])

    plt.fill_between(fill_x(my_names), means-2*(means-lowers),
                     means+2*(uppers-means), alpha=0.2)
    plt.plot(fill_x(my_names), means)
    plt.ylabel(ylabel, fontsize=46)

    noise_level = noise_levels.get(feat_name, default_noise_level)
    if X is not None:
        if noise_level > 0:
            axvlines(sorted(X[feat_name].values+np.random.uniform(
                -noise_level, noise_level, size=X[feat_name].values.shape))[::plot_every])
        else:
            axvlines(sorted(X[feat_name].values)[::plot_every])

def ebm_marginalize_feat(X, Y, feat, feat_names=None, **kwargs):
    try:
        return ebm_marginalize(X[[feat]], Y, feat_name=feat, **kwargs)
    except IndexError:
        return ebm_marginalize(
            pd.DataFrame(X[:, feat_names.index(feat)], columns=[feat]),
            Y, feat_name=feat, **kwargs)


def plot_feat(exp, feat, X=None, noise_levels={}, axlines={},
    ylims={}, xlims={}, xlabels={}, ylabel="Odds Ratio"):
    i = exp.feature_names.index(feat)
    data = exp.data(i)
    if type(data['names'][0]) is str:
        return
    fig = plt.figure(figsize=(15, 10))
    #my_min = np.min(data['scores'])
    standard_feat = standardize(feat.lower().strip())
    xlabel = xlabels.get(standard_feat, feat)
    plt.xlabel(xlabel, fontsize=54)
    if X is not None:
        my_xs = X[feat]
    for x in axlines.get(standard_feat, []):
        axline(x)
    try:
        my_names, my_xlims, lowers, uppers, means = plot_numeric(data, xlims, standard_feat)
        default_noise_level = 0.01
    except:
        my_names, my_xlims, lowers, uppers, means = plot_discrete(data, xlims, my_xs)
        default_noise_level = 0.1

    plt.xlim(my_xlims)
    my_ylims = ylims.get(standard_feat, [-0.0, 5.0])
    plt.ylim(my_ylims)
    noise_level = noise_levels.get(standard_feat, default_noise_level)
    if X is not None and noise_level > 0:
        axvlines(sorted(my_xs+np.random.uniform(-noise_level, noise_level, size=my_xs.shape))[::50])

    plt.fill_between(fill_x(my_names), means-2*(means-lowers),
                     means+2*(uppers-means), alpha=0.2)
    plt.plot(fill_x(my_names), means)
    plt.ylabel(ylabel, fontsize=46)


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
        data_dict = {
            'type': 'univariate',
            'names': names,
            'scores': impacts,
            'scores_range': (np.min(lower_bounds), np.max(upper_bounds)),
            'upper_bounds': upper_bounds,
            'lower_bounds': lower_bounds,
            'density': densities_dict,
            }
        return plot_bar(data_dict)