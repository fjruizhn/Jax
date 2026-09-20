from pathlib import Path
from policy.canonicalization.cli_v3 import main
ROOT=Path(__file__).resolve().parents[2]
def test_usage_is_64(): assert main([])==64
def test_valid_candidate_is_zero(): assert main(['candidate-hash',str(ROOT)])==0
