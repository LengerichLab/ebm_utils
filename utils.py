import numpy as np

from interpret.visual.plot import plot_bar
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl

mpl.rcParams["mpl_toolkits.legacy_colorbar"] = False
from interpret.glassbox.ebm.research.purify import purify


import matplotlib
matplotlib.rcParams.update({'font.size': 22})
def axvlines(xs, min_height=0.3, max_height=0.4, color='orange', **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = (min_height, max_height, np.nan) #plt.gca().get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims)[None, :], repeats=len(xs), axis=0).flatten()
    plot = plt.plot(x_points, y_points, color=color, alpha=0.6, scaley = False, **plot_kwargs)
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

def drop_feature(X, feat):
    try:
        X.drop(feat, axis=1, inplace=True)
    except KeyError:
        pass

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ans = [x[i] for i in range(N)]
    ans.extend((cumsum[N:] - cumsum[:-N]) / float(N))
    return ans

def add_or_append(d, key, val):
    try:
        d[key].append(val)
    except:
        d[key] = [val]

def date_to_num(date):
    day = date.split(" ")[1]
    hour = date.split(" ")[2].split(":")[0]
    return int(hour) + int(float(day))*24

def fillna_unknown_dtype_col(X):
    if X.dtype == np.int or X.dtype == np.int64:
        X = X.fillna(value=-1)
    elif X.dtype == np.float:
        X = X.fillna(value=-1)
    elif X.dtype == np.bool:
        X = X.fillna(value=False)
    elif X.dtype == np.object:
        X = X.fillna(value='missing')
    else:
        print(X.dtype)
    return X


def plot_all_day(ebm_global, vmin=-1, vmax=1, unroll_log=False, subtract_x_effect=True, model_name="", task_name="",
                 results_dir="results", cmap="bwr"):
    feat_names = ebm_global.feature_names.copy()

    for i, feat_name in enumerate(feat_names):
        data = ebm_global.data(i).copy()
        if data['type'] == 'univariate':
            feat_id = feat_names.index(feat_name)
            #mains[feat_id] = data['scores'].copy()
        elif data['type'] == 'pairwise':
            print(model_name)
            print(feat_name)
            feat_name1 = feat_name.split(' x ')[0]
            feat_name2 = feat_name.split(' x ')[1]
            left_names = np.array(data['left_names'])
            right_names = np.array(data["right_names"])
            left_good = np.isfinite(left_names)
            right_good = np.isfinite(right_names)
            mat = data['scores'].T.copy()
            left_names = left_names[left_good]
            right_names = right_names[right_good]

            # Want Day on x axis
            if feat_name1 == 'Day':
                temp = right_names.copy()
                right_names = left_names.copy()
                left_names  = temp
                mat = mat.T

                temp = feat_name1#.copy()
                feat_name1 = feat_name2#.copy()
                feat_name2 = temp

                # Add in lab value effect
                feat_id1 = feat_names.index(feat_name1)
                main1 = ebm_global.data(feat_id1)
                main_names = np.array(main1['names'])
                main_names = main_names[np.isfinite(main_names)]
                assert(np.all(main_names == left_names))
                for j in range(mat.shape[0]):
                    mat[j, :] += main1['scores'][j]

            if unroll_log:
                # Y axis to be lab value (possibly log)
                if 'log' in feat_name1:
                    feat_name1 = feat_name1.split("_log")[0]
                    left_names = 10**(left_names) - 1e-3

            # Upres
            before = mat.astype(np.float)
            left_names_left = [left_names[0]]
            left_names_left.extend([left_names[k-1] + (0.5)*(left_names[k] - left_names[k-1]) for k in range(1, len(left_names))])

            right_names_left = [right_names[0]]
            right_names_left.extend([right_names[k-1] + (0.5)*(right_names[k] - right_names[k-1]) for k in range(1, len(right_names))])

            left_names = left_names_left
            right_names = right_names_left
            left_names_upres = np.linspace(left_names[0], left_names[-1]+(left_names[-1]-left_names[-2]), 100)
            right_names_upres = np.linspace(right_names[0], right_names[-1]+(right_names[-1]-right_names[-2]), 100)
            before_upres = np.zeros((100, 100))
            for m, l_upres in enumerate(left_names_upres):
                for n, r_upres in enumerate(right_names_upres):
                    r = np.argmax(l_upres < left_names) - 1
                    s = np.argmax(r_upres < right_names) - 1
                    before_upres[m, n] = before[r, s]

            before = before_upres
            left_names = left_names_upres
            right_names = right_names_upres

            # Subtract out Day effect
            if subtract_x_effect:
                before -= np.mean(before, axis=0)

            # Subtract out Other effect
            intercept, m1, m2, after, n_iters = purify(before.copy(), densities=None)

            fig = plt.figure(figsize=(14, 5))
            grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                             nrows_ncols=(1,1),
                             axes_pad=0.15,
                             share_all=True,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="7%",
                             cbar_pad=0.15,
                             )

            extent = (right_names[0],
                      right_names[-1],
                     left_names[0],
                      left_names[-1])
            aspect = float(extent[3] - extent[2]) / float(extent[1] - extent[0])
            aspect = 1/aspect

            im = grid[0].imshow(before, aspect=aspect, origin='lower',
                interpolation='none', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
            #im = grid[1].imshow(after, aspect=aspect, origin='lower',
            #    interpolation='none', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)

            # Colorbar
            ax = grid[-1]
            ax.cax.colorbar(im)#, label="Addition")#, vmin=-1, vmax=1)
            ax.cax.set_ylabel('Addition to Mortality Log-Odds', rotation=270, fontsize=20, labelpad=20)
            ax.cax.toggle_label(True)

            grid[0].set_ylabel(feat_name1, fontsize=18)
            grid[0].set_xlabel(feat_name2, fontsize=22)
            #grid[1].set_xlabel(feat_name2, fontsize=22)
            grid[0].tick_params(axis='both', which='major', labelsize=16)
            grid[0].tick_params(axis='both', which='minor', labelsize=12)
            #grid[1].tick_params(axis='both', which='major', labelsize=16)
            #grid[1].tick_params(axis='both', which='minor', labelsize=12)
            ax.cax.tick_params(axis='both', which='major', labelsize=16)
            ax.cax.tick_params(axis='both', which='minor', labelsize=12)

            #plt.tight_layout()
            plt.savefig("{}/{}_{}_{}_{}.pdf".format(results_dir, model_name, task_name, feat_name1.replace("/", "_"), feat_name2.replace("/", "_")),
                        bbox_inches='tight', dpi=300)
            plt.show()


def plot_all(feat_names, ebm_global, mpl_style=False, figname=None, figsize=(12, 12), min_samples=None, ticksize=24):
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
                if min_samples and int(my_data['density']['scores'][1]) < min_samples:
                    print("Feature: {} does not have {} samples.".format(feat_name, min_samples))
                    continue
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
        plt.ylabel("Addition to Log Odds", fontsize=32)
        plt.yticks(fontsize=26)
        #plt.tight_layout()
        if figname is not None:
            plt.savefig(figname, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        return np.array(impacts)[sorted_i], np.array(names)[sorted_i], np.array(upper_bounds)[sorted_i]
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
