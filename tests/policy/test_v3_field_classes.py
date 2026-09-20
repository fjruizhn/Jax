from pathlib import Path
import json, shutil
import pytest
from policy.canonicalization import bootstrap_v3
from policy.canonicalization.bootstrap_v3 import bootstrap_resource_bytes, compute_bootstrap_bundle_id, verify_bootstrap_bundle
from policy.canonicalization.errors import CanonicalizationError
def test_registry_impossible_leaf_fails_even_empty_arrays(tmp_path):
 r=bootstrap_resource_bytes(); data=json.loads(r['field-classes.json']); data['fields']['authority.no_such_leaf']='NORMATIVE'; r['field-classes.json']=json.dumps(data).encode()
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)
def test_missing_registry_leaf_fails(tmp_path):
 r=bootstrap_resource_bytes(); data=json.loads(r['field-classes.json']); del data['fields']['manifest.normative_documents[*].id']; r['field-classes.json']=json.dumps(data).encode()
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)

def _repin(monkeypatch, resources):
 pin=compute_bootstrap_bundle_id(resources)
 monkeypatch.setattr(bootstrap_v3,'PINNED_BOOTSTRAP_BUNDLE_ID',pin)
 resources['BUNDLE.sha256']=(pin+'\n').encode()

def test_real_schema_new_leaf_requires_registry_entry(monkeypatch):
 r=bootstrap_resource_bytes(); schema=json.loads(r['authority-meta-contract.schema.json'])
 schema['properties']['new_normative_field']={'type':'string'}
 schema['required'].append('new_normative_field')
 r['authority-meta-contract.schema.json']=json.dumps(schema).encode(); _repin(monkeypatch,r)
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)

def test_real_schema_array_object_leaf_is_checked_when_instance_empty(monkeypatch):
 r=bootstrap_resource_bytes(); schema=json.loads(r['authoritative-policy-manifest.schema.json'])
 schema['properties']['normative_documents']['items']['properties']['new_leaf']={'type':'string'}
 schema['properties']['normative_documents']['items']['required'].append('new_leaf')
 r['authoritative-policy-manifest.schema.json']=json.dumps(schema).encode(); _repin(monkeypatch,r)
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)

def test_registry_extra_leaf_fails_with_real_schemas(monkeypatch):
 r=bootstrap_resource_bytes(); data=json.loads(r['field-classes.json'])
 data['fields']['document.impossible_field']='NORMATIVE'; r['field-classes.json']=json.dumps(data).encode(); _repin(monkeypatch,r)
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)
