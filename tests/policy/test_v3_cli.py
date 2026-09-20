from pathlib import Path
from policy.canonicalization.cli_v3 import main
import shutil
ROOT=Path(__file__).resolve().parents[2]
def test_usage_is_64(): assert main([])==64
def test_valid_candidate_is_zero(): assert main(['candidate-hash',str(ROOT)])==0
def test_malformed_candidate_is_controlled(tmp_path, capsys):
 p=tmp_path/'r'; shutil.copytree(ROOT/'policy',p/'policy')
 f=p/'policy/authority.yaml'; f.write_text(f.read_text().replace('scope: {jurisdiction: JAX, governs: [policy_corpus, authority_resolution], excludes: [runtime_authority_state, authority_ledger, decision_ledger, external_constraint_evidence]}','scope: []'))
 assert main(['candidate-hash',str(p)])==2
 assert 'Traceback' not in capsys.readouterr().err
