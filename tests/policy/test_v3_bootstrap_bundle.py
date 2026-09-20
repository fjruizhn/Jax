import pytest
from policy.canonicalization.bootstrap_v3 import bootstrap_resource_bytes,verify_bootstrap_bundle
from policy.canonicalization.errors import CanonicalizationError
@pytest.mark.parametrize('name',['BUNDLE.sha256','bundle.json','field-classes.json','authority-meta-contract.schema.json','authoritative-policy-manifest.schema.json','normative-policy-document.schema.json'])
def test_mutation_fails(name):
 r=bootstrap_resource_bytes(); r[name]+=b'\n'
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)
