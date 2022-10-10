import numpy as np
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
ebc = ExplainableBoostingClassifier
ebr = ExplainableBoostingRegressor


def fit_or_load_ebm(directory, should_refit,
                    model_name="ebm1", X_train=None, Y_train=None,
                    regression=False, **ebm_kwargs):

    if should_refit:
        if regression:
            ebm = ebr(**ebm_kwargs)
        else:
            ebm = ebc(**ebm_kwargs)
        ebm.fit(X_train, Y_train)
        np.save("{}/{}.npy".format(directory, model_name), ebm)
    else:
        ebm = np.load("{}/{}.npy".format(directory, model_name), allow_pickle=True).item()
    return ebm
