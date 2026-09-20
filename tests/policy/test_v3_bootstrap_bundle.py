import json, shutil
from pathlib import Path
import pytest
from policy.canonicalization import bootstrap_v3
from policy.canonicalization.bootstrap_v3 import bootstrap_resource_bytes,compute_bootstrap_bundle_id,verify_bootstrap_bundle
from policy.canonicalization.errors import CanonicalizationError
@pytest.mark.parametrize('name',['BUNDLE.sha256','bundle.json','field-classes.json','authority-meta-contract.schema.json','authoritative-policy-manifest.schema.json','normative-policy-document.schema.json'])
def test_mutation_fails(name):
 r=bootstrap_resource_bytes(); r[name]+=b'\n'
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)

def test_allowed_document_classes_reject_duplicates_even_with_valid_pin(monkeypatch):
 r=bootstrap_resource_bytes(); bundle=json.loads(r['bundle.json'])
 bundle['allowed_document_classes'].append('CONSTITUTIONAL_CORE')
 r['bundle.json']=json.dumps(bundle).encode()
 pin=compute_bootstrap_bundle_id(r)
 monkeypatch.setattr(bootstrap_v3,'PINNED_BOOTSTRAP_BUNDLE_ID',pin)
 r['BUNDLE.sha256']=(pin+'\n').encode()
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)

@pytest.mark.parametrize('change',[
 lambda bundle: bundle['allowed_document_classes'].pop(),
 lambda bundle: bundle['allowed_document_classes'].append('EXTRA_CLASS'),
])
def test_allowed_document_classes_require_exact_set(monkeypatch,change):
 r=bootstrap_resource_bytes(); bundle=json.loads(r['bundle.json']); change(bundle)
 r['bundle.json']=json.dumps(bundle).encode(); pin=compute_bootstrap_bundle_id(r)
 monkeypatch.setattr(bootstrap_v3,'PINNED_BOOTSTRAP_BUNDLE_ID',pin); r['BUNDLE.sha256']=(pin+'\n').encode()
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)

def test_compiled_pin_is_decisive_when_resources_and_artifact_change():
 r=bootstrap_resource_bytes(); bundle=json.loads(r['bundle.json']); bundle['purpose']='OTHER'
 r['bundle.json']=json.dumps(bundle).encode(); r['BUNDLE.sha256']=(compute_bootstrap_bundle_id(r)+'\n').encode()
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)

def test_missing_resource_is_invalid_bootstrap():
 r=bootstrap_resource_bytes(); del r['field-classes.json']
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)

def test_digest_resources_mismatch_is_invalid_even_with_fixture_pin(monkeypatch):
 r=bootstrap_resource_bytes(); bundle=json.loads(r['bundle.json']); bundle['digest_resources'].pop()
 r['bundle.json']=json.dumps(bundle).encode(); pin=compute_bootstrap_bundle_id(r)
 monkeypatch.setattr(bootstrap_v3,'PINNED_BOOTSTRAP_BUNDLE_ID',pin); r['BUNDLE.sha256']=(pin+'\n').encode()
 with pytest.raises(CanonicalizationError): verify_bootstrap_bundle(r)

def test_symlink_resource_is_rejected(tmp_path):
 source=Path(__file__).resolve().parents[2]/'policy/bootstrap/v3'; target=tmp_path/'v3'; shutil.copytree(source,target)
 (target/'field-classes.json').unlink(); (target/'field-classes.json').symlink_to(target/'bundle.json')
 with pytest.raises(CanonicalizationError): bootstrap_v3.bootstrap_resource_bytes(target)
