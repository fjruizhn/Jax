from pathlib import Path
from policy.canonicalization.strict_yaml import load_strict_yaml
def test_external_constraints_are_ceiling_only():
 x=load_strict_yaml(Path(__file__).resolve().parents[2]/'policy/authority.yaml')['external_constraints']; assert x=={'jax_normative':False,'effect':'CEILING_ONLY','may_grant_authority':False,'provenance_required_at_evaluation':True}
