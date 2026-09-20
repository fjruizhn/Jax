from pathlib import Path
import pytest
from policy.canonicalization.corpus_v3 import _safe
from policy.canonicalization.errors import CanonicalizationError
def test_non_nfc_locator_rejected(tmp_path):
 with pytest.raises(CanonicalizationError): _safe(tmp_path,'e\u0301.yaml')
