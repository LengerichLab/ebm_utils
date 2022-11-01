"""
Purification tools for EBMs.
Implementaed in:
https://github.com/blengerich/gam_purification
Install with:
pip install git+https://github.com/blengerich/gam_purification
"""
from gam_purification.models.ebm import purify_ebm as purify  # pylint: disable=unused-import
from gam_purification.models.ebm import purify_and_update, update_ebm  # pylint: disable=unused-import
