"""
Utilities for plotting learned functions.
"""

from interpret.visual.plot import plot_bar
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from plotly.offline import init_notebook_mode


EBC = ExplainableBoostingClassifier
EBR = ExplainableBoostingRegressor
init_notebook_mode(connected=True)


def axline(x_position, **kwargs):
    """Draw a single vertical line on the current plot."""
    plt.axvline(x_position, linestyle="--", color=kwargs.get("color", "gray"))


def axvlines(x_positions, **kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    x_positions = np.array(
        (x_positions,) if np.isscalar(x_positions) else x_positions, copy=False
    )
    lims = (plt.gca().get_ylim()[0], plt.gca().get_ylim()[0] + (plt.gca().get_ylim()[-1]-plt.gca().get_ylim()[0]) * 0.05, np.nan)
    x_points = np.repeat(x_positions[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(
        np.array(lims)[None, :], repeats=len(x_positions), axis=0
    ).flatten()
    plot = plt.plot(x_points, y_points,
        color=kwargs.pop("axvlines_color", "red"),
        alpha=kwargs.pop("axvlines_alpha", 0.4),
        scaley=False)
    return plot


def fill_x(orig):
    """
    Fills in the x values to make a continuous plot.
    """
    new_x = [orig[0]]
    for i in range(1, len(orig)):
        new_x.append(orig[i - 1] + (orig[i] - orig[i - 1]) / 2)
        new_x.append(orig[i - 1] + (orig[i] - orig[i - 1]) / 2)
        new_x.append(orig[i])
    return np.array(new_x)


def fill_y(orig):
    """
    Fills in the y values to make a continuous plot.
    """
    new_y = [orig[0]]
    for i in range(1, len(orig)):
        new_y.append(orig[i - 1])
        new_y.append(orig[i])
        new_y.append(orig[i])
    return np.array(new_y)


def standardize(feat):
    """
    Standardizes the string feature name.
    """
    return feat.lower().strip()


def repeat_last(np_ar):
    """
    Repeat the last value in an array.
    """
    return np.append(np_ar, np_ar[-1])


def plot_discrete(data, my_names, exp=True):
    """
    Plot effects of a discrete variable.
    """
    my_xlims = [-0.5, len(my_names) - 0.5]
    my_min = np.min([data["scores"][j] for j, x in enumerate(my_names)])
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
    """
    Plot effects of a numeric variable.
    """
    default_xlims = [np.percentile(data["names"], 2), np.percentile(data["names"], 98)]
    if xlims is None:
        my_xlims = default_xlims
    else:
        my_xlims = xlims.get(standard_feat, default_xlims)
    my_names = data["names"]
    good_name_idx = [
        np.logical_and(x >= my_xlims[0], x <= my_xlims[1]) for x in my_names[:-1]
    ]
    my_vals = [data["scores"][j] for j in range(len(good_name_idx)) if good_name_idx[j]]
    my_min = np.min(my_vals)
    if exp:
        lowers = np.exp(fill_y(repeat_last(data["lower_bounds"] - my_min)))
        uppers = np.exp(fill_y(repeat_last(data["upper_bounds"] - my_min)))
        means = np.exp(fill_y(repeat_last(data["scores"] - my_min)))
    else:
        lowers = fill_y(repeat_last(data["lower_bounds"] - my_min))
        uppers = fill_y(repeat_last(data["upper_bounds"] - my_min))
        means = fill_y(repeat_last(data["scores"] - my_min))

    return my_names, my_xlims, lowers, uppers, means


def setup_xlabel(feat, **kwargs):
    """
    Sets xlabel.
    """
    xlabel = feat
    if "xlabels" in kwargs:
        xlabel = kwargs["xlabels"].get(standardize(feat), feat)
    plt.xlabel(
        xlabel, fontsize=kwargs.get("x_fontsize", int(54 - 1.2 * (len(xlabel) // 3)))
    )


def setup_axlines(standard_feat, **kwargs):
    """
    Sets axlines.
    """
    if "axlines" in kwargs and kwargs["axlines"] is not None:
        try:
            for x_pos in kwargs["axlines"].get(standard_feat, []):
                axline(x_pos)
        except TypeError:
            try:
                for x_pos in kwargs["axlines"]:
                    axline(x_pos)
            except ValueError:
                pass


def setup_ylims(standard_feat, classification, **kwargs):
    """
    Sets y limits.
    """
    default_classification_ylims = [0.0, 5.0]
    default_regression_ylims = [0.0, 10.0]
    if classification:
        my_ylims = default_classification_ylims
    else:
        my_ylims = default_regression_ylims
    if "ylims" in kwargs and kwargs["ylims"] is not None:
        try:
            my_ylims = kwargs["ylims"][standard_feat]
        except TypeError:
            my_ylims = kwargs["ylims"]
        except KeyError:
            pass
    plt.ylim(my_ylims)


def setup_axline_hist(standard_feat, default_noise_level, my_xs, **kwargs):
    """
    Draw the axline histogram.
    """
    if "noise_level" in kwargs:
        noise_level = kwargs["noise_level"]
    elif "noise_levels" in kwargs and kwargs["noise_levels"] is not None:
        noise_level = kwargs["noise_levels"].get(standard_feat, default_noise_level)
    else:
        noise_level = default_noise_level
    if my_xs is not None:
        if noise_level > 0:
            my_xs += np.random.uniform(-noise_level, noise_level, size=my_xs.shape)
        axvlines(sorted(my_xs)[:: kwargs.pop("axvlines_every", 50)], **kwargs)


def setup_ylabel(classification, **kwargs):
    """Sets the ylabel."""
    default_classification_ylabel = "Odds Ratio"
    default_regression_ylabel = "Outcome"
    if classification:
        default_ylabel = default_classification_ylabel
    else:
        default_ylabel = default_regression_ylabel
    plt.ylabel(
        kwargs.get("ylabel", default_ylabel), fontsize=kwargs.get("y_fontsize", 46)
    )


def plot_feat(
    exp,
    feat,
    X_train=None,
    classification=True,
    **kwargs,
):
    """
    Plot the effects of a single feat.
    """

    i = exp.feature_names.index(feat)
    data = exp.data(i)
    if X_train is not None:
        my_xs = X_train[feat]
    plt.figure(figsize=kwargs.get("figsize", (15, 10)))
    setup_xlabel(feat, **kwargs)
    standard_feat = standardize(feat)
    setup_axlines(standard_feat, **kwargs)
    if exp.feature_types[i] == "continuous":
        my_names, my_xlims, lowers, uppers, means = plot_numeric(
            data, kwargs.get("xlims", None), standard_feat, classification
        )
        default_noise_level = 0.01
    elif exp.feature_types[i] == "categorical":
        my_names, my_xlims, lowers, uppers, means = plot_discrete(
            data, data["names"], classification
        )
        my_xs = np.array([float(data["names"].index(str(x))) for x in my_xs])
        default_noise_level = 0.1
    else:
        print(
            f"TODO: Don't know how to plot feature {feat} of type {exp.feature_types[i]}"
        )
        return
    plt.xlim(my_xlims)
    setup_ylims(standard_feat, classification, **kwargs)
    setup_axline_hist(standard_feat, default_noise_level, my_xs, **kwargs)

    plt.fill_between(
        fill_x(my_names),
        means - 2 * (means - lowers),
        means + 2 * (uppers - means),
        alpha=kwargs.get("fill_alpha", 0.2),
        color=kwargs.get("fill_color", "blue"),
    )
    plt.plot(fill_x(my_names), means, color=kwargs.get("mean_color", "blue"), linewidth=kwargs.get("mean_linewidth", 1))
    setup_ylabel(classification, **kwargs)
    if kwargs.get("remove_spines", False):
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def organize_all_bools(feat_names, ebm_global, **kwargs):
    """
    Organize effects of boolean variables.
    """
    names = []
    upper_bounds = []
    impacts = []
    lower_bounds = []
    densities = []
    counter = 0
    for i, feat_name in enumerate(ebm_global.feature_names):
        if feat_name in feat_names:
            my_data = ebm_global.data(i)
            if len(my_data["scores"]) == 2 and my_data["type"] == "univariate":
                n_samples = my_data["density"]["scores"][1]
                my_name = f"{feat_name} ({n_samples})"
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
                if kwargs.get("verbose", False):
                    print(
                        f"Feature: {feat_name} is not observed as a Boolean variable."
                    )
    if kwargs.get("classification", True):
        upper_bounds = np.exp(upper_bounds)
        lower_bounds = np.exp(lower_bounds)
        impacts = np.exp(impacts)
    return names, upper_bounds, impacts, lower_bounds, densities, counter


def plot_all_bools(
    feat_names,
    ebm_global,
    mpl_style=False,
    **kwargs,
):
    """
    Plot all Boolean variables.
    """
    names, upper_bounds, impacts, lower_bounds, densities, counter = organize_all_bools(
        feat_names, ebm_global, **kwargs
    )
    if counter == 0:
        return None
    if mpl_style:
        plt.figure(figsize=kwargs.get("figsize", (12, 12)))
        sorted_i = np.argsort(impacts)
        for counter, i in enumerate(sorted_i):
            plt.bar(
                counter,
                impacts[i],
                width=0.5,
                color=kwargs.get("color", "blue"),
                edgecolor=kwargs.get("edgecolor", "black"),
                yerr=np.max([0, upper_bounds[i] - impacts[i]]),
            )  # Assume symmetric error.
        plt.xticks(
            range(len(names)),
            np.array(names)[sorted_i],
            rotation=60,
            fontsize=kwargs.get("ticksize", 16),
            ha="right",
        )
        plt.ylabel(
            kwargs.get("ylabel", "Addition to Score"),
            fontsize=kwargs.get("ylabel_fontsize", 32),
        )
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        plt.yticks(fontsize=kwargs.get("ytick_fontsize", kwargs.get("ticksize", 16)))
        if "axhline" in kwargs:
            plt.axhline(
                kwargs["axhline"],
                linestyle=kwargs.get("axhline_linestyle", "--"),
                color=kwargs.get("axhline_color", "gray")
            )
        if "figname" in kwargs and kwargs["figname"] is not None:
            plt.savefig(kwargs["figname"], dpi=300, bbox_inches="tight")
        else:
            plt.show()
        return None
    # Not MPL style
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


def plot_interaction(ebm_global, mat, feat_id1, feat_id2, **kwargs):
    """Plot an interaction effect."""
    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 1),
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="10%",
        cbar_pad=0.15,
        axes_pad=0.1,
    )

    if ebm_global.feature_types[feat_id1] == "categorical":
        extent1 = (0, len(ebm_global.data(feat_id1)["names"]))
    else:
        extent1 = (
            ebm_global.data(feat_id1)["names"][0],
            ebm_global.data(feat_id1)["names"][-1],
        )
    if ebm_global.feature_types[feat_id2] == "categorical":
        extent2 = (0, len(ebm_global.data(feat_id2)["names"]))
    else:
        extent2 = (
            ebm_global.data(feat_id2)["names"][0],
            ebm_global.data(feat_id2)["names"][-1],
        )

    extent = (extent1[0], extent1[1], extent2[0], extent2[1])
    aspect = float(extent[3] - extent[2]) / float(extent[1] - extent[0])
    aspect = 1 / aspect

    image = grid[0].imshow(
        mat,
        aspect=aspect,
        origin="lower",
        interpolation="none",
        extent=extent,
        cmap="RdBu_r",
    )

    # Colorbar
    axes = grid[-1]
    axes.cax.colorbar(image)
    axes.cax.toggle_label(True)

    grid[0].set_ylabel(ebm_global.feature_names[feat_id2], fontsize=22)
    grid[0].set_xlabel(ebm_global.feature_names[feat_id1], fontsize=22)
    if ebm_global.feature_types[feat_id1] == "categorical":
        grid[0].set_xticks(
            0.5 + np.array(range(len(ebm_global.data(feat_id1)["names"]))),
            ebm_global.data(feat_id1)["names"],
        )
    if ebm_global.feature_types[feat_id2] == "categorical":
        grid[0].set_yticks(
            0.5 + np.array(range(len(ebm_global.data(feat_id2)["names"]))),
            ebm_global.data(feat_id2)["names"],
        )
    grid[0].tick_params(
        axis="both", which="major", labelsize=kwargs.get("major_tick_fontsize", 16)
    )
    grid[0].tick_params(
        axis="both", which="minor", labelsize=kwargs.get("minor_tick_fontsize", 12)
    )
    axes.cax.tick_params(
        axis="both", which="major", labelsize=kwargs.get("major_tick_fontsize", 16)
    )
    axes.cax.tick_params(
        axis="both", which="minor", labelsize=kwargs.get("minor_tick_fontsize", 12)
    )
    plt.show()


def plot_all_features(ebm_global, X_train, **kwargs):
    """Plot alll effects."""
    plot_mains(ebm_global, X_train, **kwargs)
    plot_pairs(ebm_global, **kwargs)


def plot_pairs(ebm_global, **kwargs):
    """Plot all pair effects."""
    for i, feature_name in enumerate(ebm_global.feature_names):
        if ebm_global.feature_types[i] == "interaction":
            feat_id1 = ebm_global.feature_names.index(feature_name.split(" & ")[0])
            feat_id2 = ebm_global.feature_names.index(feature_name.split(" & ")[1])
            plot_interaction(
                ebm_global, ebm_global.data(i)["scores"].T, feat_id1, feat_id2, **kwargs
            )
            plt.show()


def plot_mains(ebm_global, X_train, classification=True, **kwargs):
    """
    Plot all main effects.
    """
    if classification:
        ylabel = kwargs.pop("ylabel", "Addition to Log-Odds")
    else:
        ylabel = kwargs.pop("ylabel", "Addition to Score")
    noise_levels=kwargs.pop("noise_levels", {})
    axlines=kwargs.pop("axlines", {})
    ylims=kwargs.pop("ylims", {})
    xlims=kwargs.pop("xlims", {})
    xlabels=kwargs.pop("xlabels", {})
    
    # Plot Boolean variables on a single bar chart.
    plot_all_bools(
        ebm_global.feature_names,
        ebm_global,
        mpl_style=kwargs.pop("bool_mpl_style", True),
        figname=kwargs.pop("bool_figname", None),
        figsize=kwargs.pop("bool_figsize", (12, 12)),
        min_samples=kwargs.pop("bool_minsamples", 50),
        ticksize=kwargs.pop("bool_ticksize", 26),
        ylabel=ylabel,
        classification=classification
    )
    for i, feature_name in enumerate(ebm_global.feature_names):
        if ebm_global.feature_types[i] == "interaction":
            continue
        if len(set(X_train.values[:, i])) != 2 or kwargs.get(
            "plot_bools_individually", False
        ):
            plot_feat(
                ebm_global,
                feature_name,
                X_train,
                classification=classification,
                noise_levels=noise_levels,
                axlines=axlines,
                ylims=ylims,
                xlims=xlims,
                xlabels=xlabels,
                ylabel=ylabel,
                **kwargs
            )
            plt.show()
            if "savedir" in kwargs:
                feat_name_clean = feature_name.replace(" ", "_").replace("/", "_")
                plt.savefig(
                    f"{kwargs['savedir']}/{feat_name_clean}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )


def plot_importances(ebm, n_features=25, **kwargs):
    """
    Plot feature importances.
    Assumes the importances have already been calculated.
    """
    impts = ebm.term_importances()
    impts_sorted = list(reversed(list(sorted(enumerate(impts), key=lambda x: x[1]))))
    plt.figure(figsize=kwargs.get("figsize", (12, 6)))
    plt.bar(range(n_features), [x[1] for x in impts_sorted[:n_features]])
    plt.xticks(
        range(n_features),
        [ebm.term_names_[x[0]] for x in impts_sorted[:n_features]],
        rotation=kwargs.get("xtick_rotation", 60),
        ha="right",
        fontsize=kwargs.get("xtick_fontsize", 16),
    )
    plt.ylabel("Feature Importance", fontsize=kwargs.get("ylabel_fontsize", 32))
