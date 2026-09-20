from pathlib import Path
from policy.canonicalization.bootstrap_v3 import verified_bootstrap_v3
from policy.canonicalization.schemas_v3 import validate_manifest
from policy.canonicalization.strict_yaml import load_strict_yaml
def test_manifest_validates(): validate_manifest(load_strict_yaml(Path(__file__).resolve().parents[2]/'policy/manifest.yaml'),verified_bootstrap_v3())
