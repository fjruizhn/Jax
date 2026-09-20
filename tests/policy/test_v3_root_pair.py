from pathlib import Path
import shutil,pytest
from policy.canonicalization.corpus_v3 import validate_candidate_corpus
from policy.canonicalization.errors import CanonicalizationError
ROOT=Path(__file__).resolve().parents[2]
def test_root_id_mismatch_rejected(tmp_path):
 p=tmp_path/'r';shutil.copytree(ROOT/'policy',p/'policy');f=p/'policy/authority.yaml';f.write_text(f.read_text().replace('jax-authoritative-policy-manifest, authoritative_manifest_kind','wrong, authoritative_manifest_kind'))
 with pytest.raises(CanonicalizationError):validate_candidate_corpus(p)

@pytest.mark.parametrize('target,replacement',[
 ('authority.yaml',('jurisdiction: JAX','jurisdiction: NOT_JAX')),
 ('manifest.yaml',('jurisdiction: JAX','jurisdiction: NOT_JAX')),
 ('authority.yaml',('activation_mode: CANDIDATE_ONLY','activation_mode: ACTIVE')),
 ('manifest.yaml',('authorizes_production: false','authorizes_production: true')),
 ('authority.yaml',('kind: JAX_AUTHORITY_META_CONTRACT','kind: WRONG_KIND')),
 ('manifest.yaml',('kind: JAX_AUTHORITATIVE_POLICY_MANIFEST','kind: WRONG_KIND')),
 ('authority.yaml',('canonicalizer_version: JAX-POLICY-C14N/3','canonicalizer_version: WRONG')),
 ('authority.yaml',('authoritative_manifest_kind: JAX_AUTHORITATIVE_POLICY_MANIFEST','authoritative_manifest_kind: WRONG_KIND')),
 ('manifest.yaml',('authority_meta_contract_kind: JAX_AUTHORITY_META_CONTRACT','authority_meta_contract_kind: WRONG_KIND')),
])
def test_root_constants_rejected(tmp_path,target,replacement):
 p=tmp_path/'r'; shutil.copytree(ROOT/'policy',p/'policy'); f=p/'policy'/target
 f.write_text(f.read_text().replace(*replacement))
 with pytest.raises(CanonicalizationError): validate_candidate_corpus(p)
