import json, shutil
from pathlib import Path
import pytest
from policy.canonicalization.corpus_v3 import validate_candidate_corpus
from policy.canonicalization.errors import CanonicalizationError

ROOT=Path(__file__).resolve().parents[2]
def test_empty_membership_is_valid_candidate(): assert validate_candidate_corpus(Path(__file__).resolve().parents[2])['state']=='VALID_CANDIDATE'

def _document(identifier):
 return {
  'schema_version':'1.0','kind':'JAX_NORMATIVE_POLICY_DOCUMENT','id':identifier,
  'title':'Candidate','statement':'A binding candidate statement.',
  'scope':{'jurisdiction':'JAX','subjects':['JAX_SYSTEM'],'actions':['POLICY_EVALUATION'],'conditions_all':[]},
  'document_class':'CONSTITUTIONAL_CORE','normative_layer':'CONSTITUTIONAL_CORE',
  'lifecycle_status':'CANDIDATE','blocking':{'mode':'HARD_BLOCK'},
  'expected_enforcement':{'mode':'REQUIRED'},
  'effective_semantics':{'activation':'WHEN_CORPUS_ACTIVE','termination':'SUPERSEDED_OR_REPEALED_BY_RATIFIED_CORPUS'},
  'relationships':{'supersedes':[],'superseded_by':[]},
  'origin':{'source_type':'HUMAN_NOMINATED','source_ref':'candidate'},'notes':[],'history':[],
 }

def _candidate_with_members(tmp_path, documents):
 root=tmp_path/'r'; shutil.copytree(ROOT/'policy',root/'policy'); members=[]
 for index, document in enumerate(documents):
  filename=f"{document['id']}-{index}.yaml"
  (root/'policy'/filename).write_text(json.dumps(document))
  members.append({'id':document['id'],'path':f'policy/{filename}',
                  'document_class':document['document_class'],
                  'normative_layer':document['normative_layer'],
                  'normative_effect':'ACTIVE_WHEN_CORPUS_ACTIVE'})
 manifest=root/'policy/manifest.yaml'
 manifest.write_text(manifest.read_text().replace('normative_documents: []','normative_documents: '+json.dumps(members)))
 return root, manifest, members

def test_real_candidate_membership_keyed_set_reorder_preserves_hash(tmp_path):
 root, manifest, members=_candidate_with_members(tmp_path,[_document('doc-a'),_document('doc-b')])
 first=validate_candidate_corpus(root)['policy_corpus_hash']
 manifest.write_text(manifest.read_text().replace(json.dumps(members),json.dumps(list(reversed(members)))))
 assert validate_candidate_corpus(root)['policy_corpus_hash']==first

def test_real_candidate_membership_rejects_duplicate_normalized_id(tmp_path):
 root, _, _=_candidate_with_members(tmp_path,[_document('doc-a'),_document('doc-a')])
 with pytest.raises(CanonicalizationError,match='id duplicado|member duplicate'):
  validate_candidate_corpus(root)
