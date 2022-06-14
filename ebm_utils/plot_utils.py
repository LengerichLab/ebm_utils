
def fill_x(orig):
    new_x = [orig[0]]
    for i in range(1, len(orig)):
        new_x.append(orig[i-1] + (orig[i] - orig[i-1]) / 2)
        new_x.append(orig[i-1] + (orig[i] - orig[i-1]) / 2)
        new_x.append(orig[i])
    return np.array(new_x)


def axvlines(xs, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = (0.0, plt.gca().get_ylim()[-1]*0.05, np.nan)
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims)[None, :], repeats=len(xs), axis=0).flatten()
    plot = plt.plot(x_points, y_points, color='red', alpha=0.4, scaley = False, **plot_kwargs)
    return plot

def fill_y(orig):
    new_y = [orig[0]]
    for i in range(1, len(orig)):
        new_y.append(orig[i-1])
        new_y.append(orig[i])
        new_y.append(orig[i])
    return np.array(new_y)

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


def plot_numeric(data, xlims, standard_feat):
    my_xlims = xlims.get(standard_feat, [np.percentile(data['names'], 2), np.percentile(data['names'], 98)])
    my_names = data['names'][:-1]

    my_min = np.min([data['scores'][j] for j, x in enumerate(my_names)
                     if x <= plt.xlim()[1] and x >= plt.xlim()[0]])
    lowers = np.exp(fill_y(data['lower_bounds']-my_min))
    uppers = np.exp(fill_y(data['upper_bounds']-my_min))
    means  = np.exp(fill_y(data['scores']-my_min))
    return my_names, my_xlims, lowers, uppers, means

def ebm_marginalize(X, Y, feat_name, xlims={}, ylims={}, noise_levels={}, ylabel="Odds Ratio", plot_every=50,
                    **kwargs):
    ebm = ebc(interactions=0, **kwargs)
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
