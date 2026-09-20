from pathlib import Path
from policy.canonicalization.corpus_v3 import validate_candidate_corpus
def test_candidate_never_claims_active():
 r=validate_candidate_corpus(Path(__file__).resolve().parents[2]); assert r['state']=='VALID_CANDIDATE' and 'ACTIVE' not in r.values()
