"""
Marginalization using EBMs.
"""

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
import pandas as pd
from plotly.offline import init_notebook_mode

from ebm_utils.analysis.plot_utils import plot_feat

EBC = ExplainableBoostingClassifier
EBR = ExplainableBoostingRegressor
init_notebook_mode(connected=True)

def ebm_marginalize_col(
    x_train,
    y_train,
    feat_name,
    classification=True,
    **kwargs,
):
    """
    Use EBM to plot a marginalization of a single column in a dataframe.
    """
    if classification:
        ebm = EBC(interactions=0)
    else:
        ebm = EBR(interactions=0)
    assert x_train.shape[1] == 1  # Should only have 1 feature for marginalization.
    ebm.fit(x_train, y_train)
    exp = ebm.explain_global()
    plot_feat(
        exp,
        feat_name,
        x_train,
        noise_levels=kwargs.get("noise_levels", None),
        axlines=kwargs.get("axlines", None),
        ylims=kwargs.get("ylims", None),
        xlims=kwargs.get("xlims", None),
        xlabels=kwargs.get("xlabels", {feat_name: feat_name}),
        ylabel=kwargs.get("ylabel", "Odds Ratio"),
        classification=classification,
        **kwargs,
    )


def ebm_marginalize_feat(X_train, Y_train, feat, feat_names=None, **kwargs):
    """
    Use EBM to plot a marginalization of a single feature in a dataframe.
    """
    try:
        return ebm_marginalize_col(X_train[[feat]], Y_train, feat_name=feat, **kwargs)
    except IndexError:
        return ebm_marginalize_col(
            pd.DataFrame(X_train[:, feat_names.index(feat)], columns=[feat]),
            Y_train,
            feat_name=feat,
            **kwargs,
        )
