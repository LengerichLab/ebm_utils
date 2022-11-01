"""
Utilities files for measuring and handling feature importances.
"""
import numpy as np

from ebm_utils.analysis.embeddings import calc_embeddings


def calc_importance(ebm_global, data_np):
    """
    Calculate importance scores for features.
    """
    embeddings = calc_embeddings(ebm_global, data_np)
    return np.mean(np.abs(embeddings), axis=0)
