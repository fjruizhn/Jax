from pathlib import Path
import shutil
import pytest
from policy.canonicalization.corpus_v3 import validate_candidate_corpus
from policy.canonicalization.errors import CanonicalizationError
ROOT=Path(__file__).resolve().parents[2]
def test_duplicate_authority_sets_rejected(tmp_path):
 p=tmp_path/'r'; shutil.copytree(ROOT/'policy',p/'policy'); f=p/'policy/authority.yaml'; f.write_text(f.read_text().replace('RATIFY_CONSTITUTION, RATIFY_AMENDMENT','RATIFY_CONSTITUTION, RATIFY_CONSTITUTION'))
 with pytest.raises(CanonicalizationError): validate_candidate_corpus(p)
def test_duplicate_amendment_requirements_rejected(tmp_path):
 p=tmp_path/'r'; shutil.copytree(ROOT/'policy',p/'policy'); f=p/'policy/authority.yaml'; f.write_text(f.read_text().replace('RATIFY_AUTHORITY_META_CONTRACT_CHANGE, NEW_ROOT_PAIR','RATIFY_AUTHORITY_META_CONTRACT_CHANGE, RATIFY_AUTHORITY_META_CONTRACT_CHANGE'))
 with pytest.raises(CanonicalizationError): validate_candidate_corpus(p)

def test_set_reorder_preserves_candidate_hash(tmp_path):
 p=tmp_path/'r'; shutil.copytree(ROOT/'policy',p/'policy'); f=p/'policy/authority.yaml'
 before=validate_candidate_corpus(p)['policy_corpus_hash']
 f.write_text(f.read_text().replace('[policy_corpus, authority_resolution]','[authority_resolution, policy_corpus]'))
 assert validate_candidate_corpus(p)['policy_corpus_hash']==before

def test_ordered_precedence_reorder_changes_candidate_hash(tmp_path):
 p=tmp_path/'r'; shutil.copytree(ROOT/'policy',p/'policy'); f=p/'policy/authority.yaml'
 before=validate_candidate_corpus(p)['policy_corpus_hash']
 f.write_text(f.read_text().replace('[PROTECTED_METANORM, CONSTITUTIONAL_CORE, PRODUCT_POLICY, SUBORDINATE_POLICY]','[CONSTITUTIONAL_CORE, PROTECTED_METANORM, PRODUCT_POLICY, SUBORDINATE_POLICY]'))
 assert validate_candidate_corpus(p)['policy_corpus_hash']!=before
