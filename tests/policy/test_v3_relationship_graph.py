from policy.canonicalization.schemas_v3 import validate_document
from policy.canonicalization.corpus_v3 import validate_candidate_corpus
from policy.canonicalization.errors import CanonicalizationError
import pytest, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def doc(): return {'schema_version':'1.0','kind':'JAX_NORMATIVE_POLICY_DOCUMENT','id':'test-doc','title':'Test','statement':'Required statement','scope':{'jurisdiction':'JAX','subjects':['JAX_SYSTEM'],'actions':['POLICY_EVALUATION'],'conditions_all':[]},'document_class':'CONSTITUTIONAL_CORE','normative_layer':'CONSTITUTIONAL_CORE','lifecycle_status':'CANDIDATE','blocking':{'mode':'HARD_BLOCK'},'expected_enforcement':{'mode':'REQUIRED'},'effective_semantics':{'activation':'WHEN_CORPUS_ACTIVE','termination':'SUPERSEDED_OR_REPEALED_BY_RATIFIED_CORPUS'},'relationships':{'supersedes':[],'superseded_by':[]},'origin':{'source_type':'HUMAN_NOMINATED','source_ref':'candidate'},'notes':[],'history':[]}
def test_duplicate_edge_rejected():
 d=doc();d['relationships']['supersedes']=['other','other']
 with pytest.raises(CanonicalizationError):validate_document(d)

def _corpus_with_docs(tmp_path, docs):
 root=tmp_path/'r'; shutil.copytree(ROOT/'policy',root/'policy')
 members=[]
 for item in docs:
  name=item['id']+'.yaml'; (root/'policy'/name).write_text(json.dumps(item))
  members.append({'id':item['id'],'path':'policy/'+name,'document_class':item['document_class'],'normative_layer':item['normative_layer'],'normative_effect':'ACTIVE_WHEN_CORPUS_ACTIVE'})
 manifest=root/'policy/manifest.yaml'
 manifest.write_text(manifest.read_text().replace('normative_documents: []','normative_documents: '+json.dumps(members)))
 return root

def _named(name, supersedes=(), superseded_by=(), klass='CONSTITUTIONAL_CORE'):
 value=doc(); value['id']=name; value['document_class']=klass; value['normative_layer']=klass
 value['relationships']={'supersedes':list(supersedes),'superseded_by':list(superseded_by)}
 return value

def test_real_candidate_valid_reciprocal_graph(tmp_path):
 root=_corpus_with_docs(tmp_path,[_named('doc-a',('doc-b',)),_named('doc-b',(),('doc-a',))])
 assert validate_candidate_corpus(root)['state']=='VALID_CANDIDATE'

@pytest.mark.parametrize('docs',[
 lambda:[_named('doc-a',('doc-b',)),_named('doc-b')],
 lambda:[_named('doc-a',('doc-a',))],
 lambda:[_named('doc-a',('unknown',))],
 lambda:[_named('doc-a',('doc-b','doc-b')), _named('doc-b',(),('doc-a',))],
 lambda:[_named('doc-a',('doc-b',),('doc-b',)), _named('doc-b',('doc-a',),('doc-a',))],
 lambda:[_named('doc-a',('doc-b',)), _named('doc-b',(),('doc-a',),'PRODUCT_POLICY')],
])
def test_real_candidate_invalid_graphs(tmp_path,docs):
 root=_corpus_with_docs(tmp_path,docs())
 with pytest.raises(CanonicalizationError): validate_candidate_corpus(root)
