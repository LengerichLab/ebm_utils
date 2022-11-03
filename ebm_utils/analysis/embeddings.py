"""
Utilities files for embedding data in high-dimensional components.
"""
import numpy as np


def get_feat_vals(ebm_global, feat_id):
    """Get the bin names for a feature ID."""
    good_vals = ebm_global.data(feat_id)["names"]
    return good_vals


def find_bin(val, vals_list):
    """
    Find the first bin which has left index geq the query.
    The query belongs in the bin to the left of this one.
    Assumes the bins are right-inclusive.
    """
    try:
        idx = np.argmin(vals_list < val)
        if idx == len(vals_list) -1:
            idx -= 1
        return idx
    except TypeError:
        return np.argmin([v == val for v in vals_list])
    # comparison between bins of objects that aren't exact is not supported
    raise NotImplementedError


def calc_embeddings(ebm_global, data_np, with_uncertainties=False):
    """
    Embed data_np in the high-dimensional additive components.
    """
    preds = np.zeros((data_np.shape[0], len(ebm_global.feature_names)))
    for i in range(data_np.shape[0]):
        for j in range(data_np.shape[1]):
            good_vals = get_feat_vals(ebm_global, j)
            try:
                pred = ebm_global.data(j)["scores"][find_bin(data_np[i, j], good_vals)]
                if with_uncertainties:
                    preds[i, j] = [
                        pred,
                        ebm_global.data(j)["lower_bounds"][find_bin(data_np[i, j], good_vals)],
                        ebm_global.data(j)["upper_bounds"][find_bin(data_np[i, j], good_vals)]
                    ]
                else:
                    preds[i, j] = pred
            except KeyError:
                pass

            for k in range(j, data_np.shape[1]):  # j < k
                pair_feat_name = (
                    f"{ebm_global.feature_names[j]} data_np {ebm_global.feature_names[k]}"
                )
                if pair_feat_name not in ebm_global.feature_names:
                    continue
                feat_idx = ebm_global.feature_names.index(pair_feat_name)
                good_vals2 = get_feat_vals(ebm_global, k)
                idx1 = find_bin(data_np[i, j], good_vals)
                idx2 = find_bin(data_np[i, k], good_vals2)
                pred = ebm_global.data(feat_idx)["scores"][idx1, idx2]
                if with_uncertainties:
                    preds[i, feat_idx] = [
                        pred,
                        ebm_global.data(feat_idx)["lower_bounds"][idx1, idx2],
                        ebm_global.data(feat_idx)["upper_bounds"][idx1, idx2]
                    ]
                else:
                    preds[i, feat_idx] = pred
    return preds
