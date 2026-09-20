from pathlib import Path
from policy.canonicalization.corpus_v3 import validate_candidate_corpus
def test_candidate_has_hash(): assert validate_candidate_corpus(Path(__file__).resolve().parents[2])['policy_corpus_hash'].startswith('sha256:')
