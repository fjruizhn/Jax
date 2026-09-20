from policy.canonicalization.bootstrap_v3 import V3_RESOURCE_NAMES
def test_exact_root_resources(): assert V3_RESOURCE_NAMES == ('bundle.json','field-classes.json','authority-meta-contract.schema.json','authoritative-policy-manifest.schema.json','normative-policy-document.schema.json')
