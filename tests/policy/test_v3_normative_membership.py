from pathlib import Path
from policy.canonicalization.corpus_v3 import validate_candidate_corpus
def test_empty_membership_is_valid_candidate(): assert validate_candidate_corpus(Path(__file__).resolve().parents[2])['state']=='VALID_CANDIDATE'
