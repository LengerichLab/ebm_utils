from interpret.visual.plot import plot_bar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
ebc = ExplainableBoostingClassifier
ebr = ExplainableBoostingRegressor

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)


def axvlines(xs, **kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    xs = np.array((xs,) if np.isscalar(xs) else xs, copy=False)
    lims = (0.0, plt.gca().get_ylim()[-1] * 0.05, np.nan)
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims)[None, :], repeats=len(xs), axis=0).flatten()
    plot = plt.plot(x_points, y_points, color="red", alpha=0.4, scaley=False, **kwargs)
    return plot


def fill_x(orig):
    new_x = [orig[0]]
    for i in range(1, len(orig)):
        new_x.append(orig[i - 1] + (orig[i] - orig[i - 1]) / 2)
        new_x.append(orig[i - 1] + (orig[i] - orig[i - 1]) / 2)
        new_x.append(orig[i])
    return np.array(new_x)


def fill_y(orig):
    new_y = [orig[0]]
    for i in range(1, len(orig)):
        new_y.append(orig[i - 1])
        new_y.append(orig[i])
        new_y.append(orig[i])
    return np.array(new_y)


def axline(x):
    plt.axvline(x, linestyle="--", color="gray")


def standardize(feat):
    # TODO
    return feat


def repeat_last(np_ar):
    return np.append(np_ar, np_ar[-1])


def plot_discrete(data, xlims, my_names, exp=True):
    my_xlims = [-0.5, len(my_names) - 0.5]
    my_min = np.min(
        [
            data["scores"][j] for j, x in enumerate(my_names)
        ]
    )
    if exp:
        lowers = np.exp(fill_y(data["lower_bounds"] - my_min))
        uppers = np.exp(fill_y(data["upper_bounds"] - my_min))
        means = np.exp(fill_y(data["scores"] - my_min))
    else:
        lowers = fill_y(data["lower_bounds"] - my_min)
        uppers = fill_y(data["upper_bounds"] - my_min)
        means = fill_y(data["scores"] - my_min)
    if len(my_names) > 5 or np.max([len(n) for n in my_names]) > 10:
        plt.xticks(range(len(my_names)), my_names, rotation=60, ha="right")
    else:
        plt.xticks(range(len(my_names)), my_names)

    my_names_as_ints = np.array(list(range(len(my_names))))
    return my_names_as_ints, my_xlims, lowers, uppers, means


def plot_numeric(data, xlims, standard_feat, exp=True):
    my_xlims = xlims.get(
        standard_feat,
        [np.percentile(data["names"], 2), np.percentile(data["names"], 98)],
    )
    my_names = data["names"]
    my_min = np.min(
        [
            repeat_last(data["scores"])[j]
            for j, x in enumerate(my_names)
            if x >= my_xlims[0] and x <= my_xlims[1]
        ]
    )
    if exp:
        lowers = np.exp(fill_y(repeat_last(data["lower_bounds"] - my_min)))
        uppers = np.exp(fill_y(repeat_last(data["upper_bounds"] - my_min)))
        means = np.exp(fill_y(repeat_last(data["scores"] - my_min)))
    else:
        lowers = fill_y(repeat_last(data["lower_bounds"] - my_min))
        uppers = fill_y(repeat_last(data["upper_bounds"] - my_min))
        means = fill_y(repeat_last(data["scores"] - my_min))

    return my_names, my_xlims, lowers, uppers, means


def plot_feat(
    exp,
    feat,
    X=None,
    noise_levels={},
    axlines={},
    ylims={},
    xlims={},
    xlabels={},
    ylabel="Odds Ratio",
    classification=True,
    **kwargs
):
    i = exp.feature_names.index(feat)
    data = exp.data(i)
    #if type(data["names"][0]) is str:
    #    return
    fig = plt.figure(figsize=kwargs.get('figsize', (15, 10)))
    standard_feat = standardize(feat.lower().strip())
    xlabel = xlabels.get(standard_feat, feat)
    plt.xlabel(xlabel,
        fontsize=kwargs.get('x_fontsize', int(54 - 1.2*(len(xlabel) // 3))))
    if X is not None:
        my_xs = X[feat]
    for x in axlines.get(standard_feat, []):
        axline(x)
    if exp.feature_types[i] == 'continuous':
        my_names, my_xlims, lowers, uppers, means = plot_numeric(
            data, xlims, standard_feat, classification
        )
        default_noise_level = 0.01
    elif exp.feature_types[i] == 'categorical':
        my_names, my_xlims, lowers, uppers, means = plot_discrete(
            data, xlims, data['names'], classification)
        my_xs = np.array([float(data['names'].index(str(x))) for x in my_xs])
        default_noise_level = 0.1
    else:
        print("TODO: Don't know how to plot feature {} of type {}".format(feat, exp.feature_types[i]))
        return

    plt.xlim(my_xlims)
    my_ylims = ylims.get(standard_feat, [-0.0, 5.0])
    plt.ylim(my_ylims)
    noise_level = noise_levels.get(standard_feat, default_noise_level)
    if X is not None and noise_level > 0:
        axvlines(
            sorted(
                my_xs + np.random.uniform(-noise_level, noise_level, size=my_xs.shape)
            )[::kwargs.get('axvlines_every', 50)]
        )

    plt.fill_between(
        fill_x(my_names),
        means - 2 * (means - lowers),
        means + 2 * (uppers - means),
        alpha=0.2,
    )
    plt.plot(fill_x(my_names), means)
    plt.ylabel(ylabel, fontsize=kwargs.get('y_fontsize', 46))


def plot_all_bools(
    feat_names,
    ebm_global,
    mpl_style=False,
    figname=None,
    figsize=(12, 12),
    min_samples=None,
    ticksize=26,
    verbose=False,
    **kwargs
):
    names = []
    upper_bounds = []
    impacts = []
    lower_bounds = []
    densities = []
    counter = 0
    for i, feat_name in enumerate(ebm_global.feature_names):
        if feat_name in feat_names:
            my_data = ebm_global.data(i)
            if len(my_data["scores"]) == 2 and my_data['type'] == 'univariate':
                my_name = "{} ({})".format(feat_name, my_data["density"]["scores"][1])
                names.append(my_name)
                impacts.append(my_data["scores"][1] - my_data["scores"][0])
                upper_bounds.append(
                    my_data["upper_bounds"][1] - my_data["lower_bounds"][0]
                )
                lower_bounds.append(
                    my_data["lower_bounds"][1] - my_data["upper_bounds"][0]
                )
                densities.append(my_data["density"]["scores"][1])
                counter += 1
            else:
                if verbose:
                    print("Feature: {} is not observed as a Boolean variable.".format(
                        feat_name))
    if mpl_style:
        fig = plt.figure(figsize=figsize)
        sorted_i = np.argsort(impacts)
        for counter, i in enumerate(sorted_i):
            plt.bar(
                counter,
                impacts[i],
                width=0.5,
                color="blue",
                edgecolor="black",
                yerr=np.max([0, upper_bounds[i] - impacts[i]]),
            )  # Assume symmetric error.
        plt.xticks(
            range(len(names)), np.array(names)[sorted_i], rotation=60, fontsize=ticksize, ha='right'
        )
        plt.ylabel(kwargs.get("ylabel", "Addition to Score"),
            fontsize=kwargs.get('ylabel_fontsize', 32))
        plt.yticks(fontsize=kwargs.get('ytick_fontsize', ticksize))
        if figname is not None:
            plt.savefig(figname, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    else:
        sorted_i = np.argsort(impacts)
        names = np.array(names)[sorted_i]
        impacts = np.array(impacts)[sorted_i]
        upper_bounds = np.array(upper_bounds)[sorted_i]
        lower_bounds = np.array(lower_bounds)[sorted_i]
        densities_dict = {"names": names, "scores": np.array(densities)[sorted_i]}
        data_dict = {
            "type": "univariate",
            "names": names,
            "scores": impacts,
            "scores_range": (np.min(lower_bounds), np.max(upper_bounds)),
            "upper_bounds": upper_bounds,
            "lower_bounds": lower_bounds,
            "density": densities_dict,
        }
        return plot_bar(data_dict)


def ebm_marginalize(
    X,
    Y,
    feat_name,
    xlims={},
    ylims={},
    noise_levels={},
    ylabel="Odds Ratio",
    xlabels={},
    classification=True,
    **kwargs,
):
    if classification:
        ebm = ebc(interactions=0)
    else:
        ebm = ebr(interactions=0)
    assert X.shape[1] == 1  # Should only have 1 feature for marginalization.
    ebm.fit(X, Y)
    exp = ebm.explain_global()
    plot_feat(exp, feat_name, X, noise_levels, kwargs.get('axlines', {}),
        ylims, xlims, xlabels, ylabel, classification=classification, **kwargs)

    """
    data = exp.data(0)
    fig = plt.figure(figsize=(15, 10))
    plt.xlabel(feat_name, fontsize=54)
    try:
        my_names, my_xlims, lowers, uppers, means = plot_numeric(
            data, xlims, feat_name, exp=classification
        )
        default_noise_level = 0.01
    except:
        my_names, my_xlims, lowers, uppers, means = plot_discrete(
            data, xlims, list(set(X[feat_name].values)), exp=classification
        )
        default_noise_level = 0.1
    plt.xlim(my_xlims)
    if feat_name in ylims:
        plt.ylim(ylims[feat_name])

    plt.fill_between(
        fill_x(my_names),
        means - 2 * (means - lowers),
        means + 2 * (uppers - means),
        alpha=0.2,
    )
    plt.plot(fill_x(my_names), means)
    plt.ylabel(ylabel, fontsize=46)

    noise_level = noise_levels.get(feat_name, default_noise_level)
    if X is not None:
        if noise_level > 0:
            axvlines(
                sorted(
                    X[feat_name].values
                    + np.random.uniform(
                        -noise_level, noise_level, size=X[feat_name].values.shape
                    )
                )[::plot_every]
            )
        else:
            axvlines(sorted(X[feat_name].values)[::plot_every])
    """


def ebm_marginalize_feat(X, Y, feat, feat_names=None, **kwargs):
    try:
        return ebm_marginalize(X[[feat]], Y, feat_name=feat, **kwargs)
    except IndexError:
        return ebm_marginalize(
            pd.DataFrame(X[:, feat_names.index(feat)], columns=[feat]),
            Y,
            feat_name=feat,
            **kwargs,
        )


from mpl_toolkits.axes_grid1 import ImageGrid
def plot_interaction(ebm_global, mat, feat_id1, feat_id2):
    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 1),
                     share_all=True,
                     cbar_location='right',
                     cbar_mode='single',
                     cbar_size='10%',
                     cbar_pad=0.15,
                     axes_pad=0.1)

    if ebm_global.feature_types[feat_id1] == 'categorical':
        extent1 = (0, len(ebm_global.data(feat_id1)['names']))
    else:
        extent1 = (ebm_global.data(feat_id1)['names'][0],
              ebm_global.data(feat_id1)['names'][-1])
    if ebm_global.feature_types[feat_id2] == 'categorical':
        extent2 = (0, len(ebm_global.data(feat_id2)['names']))
    else:
        extent2 = (ebm_global.data(feat_id2)['names'][0],
              ebm_global.data(feat_id2)['names'][-1])

    extent = (extent1[0], extent1[1], extent2[0], extent2[1])
    aspect = float(extent[3] - extent[2]) / float(extent[1] - extent[0])
    aspect = 1/aspect

    im = grid[0].imshow(mat, aspect=aspect, origin='lower',
                        interpolation='none', extent=extent, cmap='RdBu_r')

    # Colorbar
    ax = grid[-1]
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    grid[0].set_ylabel(ebm_global.feature_names[feat_id2], fontsize=22)
    grid[0].set_xlabel(ebm_global.feature_names[feat_id1], fontsize=22)
    if ebm_global.feature_types[feat_id1] == 'categorical':
        grid[0].set_xticks(0.5+np.array(range(len(ebm_global.data(feat_id1)['names']))),
            ebm_global.data(feat_id1)['names'])
    if ebm_global.feature_types[feat_id2] == 'categorical':
        grid[0].set_yticks(0.5+np.array(range(len(ebm_global.data(feat_id2)['names']))),
            ebm_global.data(feat_id2)['names'])
    grid[0].tick_params(axis='both', which='major', labelsize=16)
    grid[0].tick_params(axis='both', which='minor', labelsize=12)
    ax.cax.tick_params(axis='both', which='major', labelsize=16)
    ax.cax.tick_params(axis='both', which='minor', labelsize=12)
    plt.show()


def plot_all_features(ebm_global, X_train, **kwargs):
    plot_mains(ebm_global, X_train, **kwargs)
    plot_pairs(ebm_global, X_train, **kwargs)


def plot_pairs(ebm_global, X_train, **kwargs):
    for i, feature_name in enumerate(ebm_global.feature_names):
        if ebm_global.feature_types[i] == 'interaction':
            feat_id1 = ebm_global.feature_names.index(feature_name.split(' x ')[0])
            feat_id2 = ebm_global.feature_names.index(feature_name.split(' x ')[1])
            plot_interaction(ebm_global, ebm_global.data(i)['scores'].T, feat_id1, feat_id2)
            plt.show()


def plot_mains(ebm_global, X_train, **kwargs):
    plot_all_bools(ebm_global.feature_names, ebm_global,
        mpl_style=kwargs.get('bool_mpl_style'),
        figname=kwargs.get('bool_figname', None),
        figsize=kwargs.get('bool_figsize', (12, 12)),
        min_samples=kwargs.get('bool_minsamples', 50),
        ticksize=kwargs.get('bool_ticksize', 26),
        ylabel=kwargs.get('ylabel', 'Addition to Score'))
    for i, feature_name in enumerate(ebm_global.feature_names):
        if ebm_global.feature_types[i] != 'interaction':
            plot_feat(
                ebm_global, feature_name, X_train,
                noise_levels=kwargs.get('noise_levels', {}),
                axlines=kwargs.get('axlines', {}),
                ylims=kwargs.get('ylims', {}),
                xlims=kwargs.get('xlims', {}),
                xlabels=kwargs.get('xlabels', {}),
                ylabel=kwargs.get('ylabel', 'Addition to Score'))
            plt.show()
            if 'savedir' in kwargs:
                plt.savefig("{}/{}.pdf".format(kwargs['savedir'], feature_name), dpi=300)


def plot_importances(ebm_global, n_features=25):
    impts = ebm_global.feature_importances_
    impts_sorted = list(reversed(list(sorted(enumerate(impts), key=lambda x: x[1]))))
    fig = plt.figure(figsize=(12, 6))
    plt.bar(range(n_features), [x[1] for x in impts_sorted[:n_features]])
    plt.xticks(range(n_features), [ebm_global.feature_names[x[0]] for x in impts_sorted[:n_features]],
              rotation=60, ha='right', fontsize=16)
    plt.ylabel("Feature Importance", fontsize=32)

"""
def plot_mains(
    key, ebm_global, X_stds, X_means, ebm_results, xgb_results, dataset_name
):
    # TODO: Center
    feat_vals = np.array(get_feat_vals(ebm_global, key))
    if X_stds is not None:
        feat_vals *= X_stds[key]
        feat_vals += X_means[key]

    my_zero = np.zeros_like(ebm_results[1][0][key])
    all_ebm_mains = [x[0].get(key, my_zero) for x in ebm_results]
    all_xgb_mains = [x[0].get(key, my_zero) for x in xgb_results]

    y_min = np.min(np.vstack((np.array(all_xgb_mains), np.array(all_ebm_mains))))
    y_min -= 0.05 * np.abs(y_min)
    y_max = np.max(np.vstack((np.array(all_xgb_mains), np.array(all_ebm_mains))))
    y_max += 0.05 * np.abs(y_max)

    fig = plt.figure(figsize=(14, 10))
    ax = plt.subplot(1, 2, 1)
    ax.set_title("XGB-2", fontsize=20)
    colors = ["black", "red", "blue", "orange", "teal", "green"]
    for i, x in enumerate(all_xgb_mains):
        plt.plot(feat_vals, x, linestyle="--", color=colors[i], label=xgb_results[i][1])

    ax.set_ylabel("Addition to Score", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim([y_min, y_max])
    plt.xlabel(ebm_global.feature_names[key], fontsize=18)

    ax = plt.subplot(1, 2, 2)
    ax.set_title("GA2M", fontsize=20)
    for i, x in enumerate(all_ebm_mains):
        plt.plot(feat_vals, x, linestyle="--", color=colors[i], label=ebm_results[i][1])

    lgd = plt.legend(fontsize=16, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim([y_min, y_max])

    plt.xlabel(ebm_global.feature_names[key], fontsize=18)
    plt.savefig(
        "figs/{}/{}.pdf".format(dataset_name, ebm_global.feature_names[key]),
        dpi=300,
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
"""