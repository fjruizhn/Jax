from pathlib import Path
import pytest
from policy.canonicalization.corpus_v3 import _safe
from policy.canonicalization.errors import CanonicalizationError
@pytest.mark.parametrize('raw',['../x','./x','/x','x/*','x\\y'])
def test_unsafe_locators_rejected(tmp_path,raw):
 with pytest.raises(CanonicalizationError):_safe(tmp_path,raw)
