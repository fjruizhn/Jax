from pathlib import Path
import shutil,pytest
from policy.canonicalization.corpus_v3 import validate_candidate_corpus
from policy.canonicalization.errors import CanonicalizationError
ROOT=Path(__file__).resolve().parents[2]
def test_duplicate_reference_locator_rejected(tmp_path):
 p=tmp_path/'r';shutil.copytree(ROOT/'policy',p/'policy'); f=p/'policy/manifest.yaml';f.write_text(f.read_text().replace('reference_documents:\n','reference_documents:\n  - {reference_id: legacy-p10-copy, source_locator: policy/rules/P10-fail-open-prohibido.yaml, source_type: LEGACY_CORPUS, legacy_status: NORMATIVA, display_name: copy}\n'))
 with pytest.raises(CanonicalizationError):validate_candidate_corpus(p)
