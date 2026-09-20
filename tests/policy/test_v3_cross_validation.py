from pathlib import Path
import shutil
import pytest
from policy.canonicalization.corpus_v3 import validate_candidate_corpus
from policy.canonicalization.errors import CanonicalizationError
ROOT=Path(__file__).resolve().parents[2]
def repo(tmp_path):
 p=tmp_path/'repo'; shutil.copytree(ROOT/'policy',p/'policy'); return p
@pytest.mark.parametrize('path,old,new',[('policy/authority.yaml','jurisdiction: JAX','jurisdiction: NOT_JAX'),('policy/manifest.yaml','jurisdiction: JAX','jurisdiction: NOT_JAX')])
def test_jurisdiction_is_constant(tmp_path,path,old,new):
 p=repo(tmp_path); f=p/path; f.write_text(f.read_text().replace(old,new));
 with pytest.raises(CanonicalizationError): validate_candidate_corpus(p)
def test_both_non_jax_rejected(tmp_path):
 p=repo(tmp_path)
 for n in ('authority.yaml','manifest.yaml'):
  f=p/'policy'/n; f.write_text(f.read_text().replace('jurisdiction: JAX','jurisdiction: NOT_JAX'))
 with pytest.raises(CanonicalizationError): validate_candidate_corpus(p)
