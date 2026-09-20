from pathlib import Path
import json, shutil
import pytest
from policy.canonicalization.bootstrap_v3 import bootstrap_resource_bytes, verify_bootstrap_bundle
from policy.canonicalization.errors import CanonicalizationError
def test_registry_impossible_leaf_fails_even_empty_arrays(tmp_path):
 r=bootstrap_resource_bytes(); data=json.loads(r['field-classes.json']); data['fields']['authority.no_such_leaf']='NORMATIVE'; r['field-classes.json']=json.dumps(data).encode()
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)
def test_missing_registry_leaf_fails(tmp_path):
 r=bootstrap_resource_bytes(); data=json.loads(r['field-classes.json']); del data['fields']['manifest.normative_documents[*].id']; r['field-classes.json']=json.dumps(data).encode()
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)
