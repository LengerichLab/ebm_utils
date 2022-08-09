from interpret.visual.plot import plot_bar
import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import init_notebook_mode
import pandas as pd
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)

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


def plot_discrete(data, xlims, my_names, exp=True):
    my_xlims = [-0.5, len(my_names) - 0.5]
    my_min = np.min(
        [
            data["scores"][j]
            for j, x in enumerate(my_names)
            if x <= plt.xlim()[1] and x >= plt.xlim()[0]
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
    if len(my_names) > 5:
        plt.xticks(rotation=60, ha="right")
    return my_names, my_xlims, lowers, uppers, means


def repeat_last(np_ar):
    return np.append(np_ar, np_ar[-1])


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
            if x <= plt.xlim()[1] and x >= plt.xlim()[0]
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


def ebm_marginalize(
    X,
    Y,
    feat_name,
    xlims={},
    ylims={},
    noise_levels={},
    ylabel="Odds Ratio",
    plot_every=50,
    classification=True,
    **kwargs,
):
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
):
    i = exp.feature_names.index(feat)
    data = exp.data(i)
    if type(data["names"][0]) is str:
        return
    fig = plt.figure(figsize=(15, 10))
    # my_min = np.min(data['scores'])
    standard_feat = standardize(feat.lower().strip())
    xlabel = xlabels.get(standard_feat, feat)
    plt.xlabel(xlabel, fontsize=54)
    if X is not None:
        my_xs = X[feat]
    for x in axlines.get(standard_feat, []):
        axline(x)
    try:
        my_names, my_xlims, lowers, uppers, means = plot_numeric(
            data, xlims, standard_feat
        )
        default_noise_level = 0.01
    except:
        my_names, my_xlims, lowers, uppers, means = plot_discrete(data, xlims, my_xs)
        default_noise_level = 0.1

    plt.xlim(my_xlims)
    my_ylims = ylims.get(standard_feat, [-0.0, 5.0])
    plt.ylim(my_ylims)
    noise_level = noise_levels.get(standard_feat, default_noise_level)
    if X is not None and noise_level > 0:
        axvlines(
            sorted(
                my_xs + np.random.uniform(-noise_level, noise_level, size=my_xs.shape)
            )[::50]
        )

    plt.fill_between(
        fill_x(my_names),
        means - 2 * (means - lowers),
        means + 2 * (uppers - means),
        alpha=0.2,
    )
    plt.plot(fill_x(my_names), means)
    plt.ylabel(ylabel, fontsize=46)


def plot_all_bools(
    feat_names,
    ebm_global,
    mpl_style=False,
    figname=None,
    figsize=(12, 12),
    min_samples=None,
    ticksize=26,
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
            if len(my_data["scores"]) == 2:
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
                print(
                    "Feature: {} is not observed as a Boolean variable.".format(
                        feat_name
                    )
                )
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
                yerr=upper_bounds[i] - impacts[i],
            )  # Assume symmetric error.
        plt.xticks(
            range(len(names)), np.array(names)[sorted_i], rotation=90, fontsize=ticksize
        )
        plt.ylabel("Addition to Score", fontsize=32)
        plt.yticks(fontsize=ticksize)
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


def fit_or_load_ebm(
    directory,
    should_refit,
    X_train=None,
    y_train=None,
    n_estimators=100,
    feature_step_n_inner_bags=100,
    regression=False,
    min_cases_for_splits=25,
    early_stopping_tolerance=1e-5,
    early_stopping_run_length=25,
):

    if should_refit:
        if regression:
            ebm1 = ExplainableBoostingRegressor(
                interactions=0,
                n_estimators=n_estimators,
                feature_step_n_inner_bags=feature_step_n_inner_bags,
                min_cases_for_splits=min_cases_for_splits,
                early_stopping_tolerance=early_stopping_tolerance,
                early_stopping_run_length=early_stopping_run_length,
            )
        else:
            ebm1 = ExplainableBoostingClassifier(
                interactions=0,
                n_estimators=n_estimators,
                feature_step_n_inner_bags=feature_step_n_inner_bags,
                min_cases_for_splits=min_cases_for_splits,
                early_stopping_tolerance=early_stopping_tolerance,
                early_stopping_run_length=early_stopping_run_length,
            )

        ebm1.fit(X_train, y_train)
        np.save("{}/ebm1.npy".format(directory), ebm1)

        if regression:
            ebm = ExplainableBoostingRegressor(
                interactions=50,
                n_estimators=n_estimators,
                feature_step_n_inner_bags=feature_step_n_inner_bags,
            )
        else:
            ebm = ExplainableBoostingClassifier(
                interactions=50,
                n_estimators=n_estimators,
                feature_step_n_inner_bags=feature_step_n_inner_bags,
            )
        ebm.fit(X_train, y_train)
        np.save("{}/ebm.npy".format(directory), ebm)
    else:
        ebm1 = np.load("{}/ebm1.npy".format(directory), allow_pickle=True).item()
        ebm = np.load("{}/ebm.npy".format(directory), allow_pickle=True).item()
    return ebm, ebm1


def make_preds(ebm_global, X, margs, pairs, intercept=0):
    preds = []
    preds_mains = []
    pairs_keys_set = set(pairs.keys())
    for i in range(X.shape[0]):
        if i % 100 == 0:
            print(i, end="\r")
        pred = 0
        pred_main = 0
        for j in range(X.shape[1]):
            good_vals = get_feat_vals(ebm_global, j)
            try:
                pred_main += margs[j][find_bin(X[i, j], good_vals)]
            except KeyError:
                pass

            for k in range(j, X.shape[1]):  # j < k
                if (j, k) not in pairs_keys_set:
                    continue
                good_vals2 = get_feat_vals(ebm_global, k)
                idx1 = find_bin(X[i, j], good_vals)
                idx2 = find_bin(X[i, k], good_vals2)
                pred += pairs[(j, k)][idx1, idx2]
        preds.append(pred_main + pred)
        preds_mains.append(pred_main)
    return np.array(preds) + intercept, np.array(preds_mains) + intercept


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
