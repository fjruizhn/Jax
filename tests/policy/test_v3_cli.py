from pathlib import Path
from policy.canonicalization.cli_v3 import main
import pytest, shutil
ROOT=Path(__file__).resolve().parents[2]
def test_usage_is_64(): assert main([])==64
def test_valid_candidate_is_zero(): assert main(['candidate-hash',str(ROOT)])==0
def test_malformed_candidate_is_controlled(tmp_path, capsys):
 p=tmp_path/'r'; shutil.copytree(ROOT/'policy',p/'policy')
 f=p/'policy/authority.yaml'; f.write_text(f.read_text().replace('scope: {jurisdiction: JAX, governs: [policy_corpus, authority_resolution], excludes: [runtime_authority_state, authority_ledger, decision_ledger, external_constraint_evidence]}','scope: []'))
 assert main(['candidate-hash',str(p)])==2
 assert 'Traceback' not in capsys.readouterr().err

@pytest.mark.parametrize('relative_path,old,new',[
 ('policy/authority.yaml',
  'scope: {jurisdiction: JAX, governs: [policy_corpus, authority_resolution], excludes: [runtime_authority_state, authority_ledger, decision_ledger, external_constraint_evidence]}',
  'scope: []'),
 ('policy/authority.yaml',
  'scope: {jurisdiction: JAX, governs: [policy_corpus, authority_resolution], excludes: [runtime_authority_state, authority_ledger, decision_ledger, external_constraint_evidence]}',
  'scope: malformed'),
 ('policy/authority.yaml','governs: [policy_corpus, authority_resolution]','governs: {not: an-array}'),
 ('policy/authority.yaml','governs: [policy_corpus, authority_resolution]','governs: scalar'),
 ('policy/manifest.yaml','normative_documents: []','normative_documents: [malformed-member]'),
])
def test_cli_malformed_shape_matrix_is_controlled(tmp_path,capsys,relative_path,old,new):
 p=tmp_path/'r'; shutil.copytree(ROOT/'policy',p/'policy')
 target=p/relative_path; text=target.read_text(); assert old in text
 target.write_text(text.replace(old,new))
 assert main(['candidate-hash',str(p)])==2
 assert 'Traceback' not in capsys.readouterr().err
