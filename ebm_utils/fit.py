"""
Helper functions to fit EBMs.
"""

from interpret.glassbox import ExplainableBoostingClassifier as ebc


def fit_ebm(X_train, Y_train, ebm_constructor_kwargs=None, ebm_fit_kwargs=None):
    """
    Helper function to fit an EBM to a dataset.
    Returns the fitted EBM.
    """
    if ebm_constructor_kwargs is not None:
        ebm = ebc(**ebm_constructor_kwargs)
    else:
        ebm = ebc()
    if ebm_fit_kwargs is not None:
        ebm.fit(X_train, Y_train, **ebm_fit_kwargs)
    else:
        ebm.fit(X_train, Y_train)
    return ebm
