from pathlib import Path
import shutil,pytest
from policy.canonicalization.corpus_v3 import validate_candidate_corpus
from policy.canonicalization.errors import CanonicalizationError
ROOT=Path(__file__).resolve().parents[2]
def test_root_id_mismatch_rejected(tmp_path):
 p=tmp_path/'r';shutil.copytree(ROOT/'policy',p/'policy');f=p/'policy/authority.yaml';f.write_text(f.read_text().replace('jax-authoritative-policy-manifest, authoritative_manifest_kind','wrong, authoritative_manifest_kind'))
 with pytest.raises(CanonicalizationError):validate_candidate_corpus(p)
