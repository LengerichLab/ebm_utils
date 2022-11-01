"""
Utility for saving and loading EBMs.
"""

import numpy as np
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
EBC = ExplainableBoostingClassifier
EBR = ExplainableBoostingRegressor


def load_ebm(directory, model_name):
    """
    Loads and ebm from provided directory and filename.
    """
    return np.load(f"{directory}/{model_name}.npy", allow_pickle=True).item()


def fit_or_load_ebm(directory, should_refit,
                    model_name="ebm1", data=None,
                    regression=False, **ebm_kwargs):
    """
    Fits and saves, or loads,
    an EBM with save/load depending on the should_refit parameter.
    """

    if should_refit:
        if regression:
            ebm = EBR(**ebm_kwargs)
        else:
            ebm = EBC(**ebm_kwargs)
        ebm.fit(data["X_train"], data["Y_train"])
        np.save(f"{directory}/{model_name}.npy", ebm)
        return ebm
    return load_ebm(directory, model_name)
