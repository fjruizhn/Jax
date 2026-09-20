from pathlib import Path
from policy.canonicalization.strict_yaml import load_strict_yaml
def test_protected_metanorms_inline_only():
 a=load_strict_yaml(Path(__file__).resolve().parents[2]/'policy/authority.yaml'); assert len(a['protected_metanorms'])==5
