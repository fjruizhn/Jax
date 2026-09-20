import copy
import pytest
from policy.canonicalization.schemas_v3 import validate_document
from policy.canonicalization.errors import CanonicalizationError
def doc(): return {'schema_version':'1.0','kind':'JAX_NORMATIVE_POLICY_DOCUMENT','id':'test-doc','title':'Test','statement':'Required statement','scope':{'jurisdiction':'JAX','subjects':['JAX_SYSTEM'],'actions':['POLICY_EVALUATION'],'conditions_all':[]},'document_class':'CONSTITUTIONAL_CORE','normative_layer':'CONSTITUTIONAL_CORE','lifecycle_status':'CANDIDATE','blocking':{'mode':'HARD_BLOCK'},'expected_enforcement':{'mode':'REQUIRED'},'effective_semantics':{'activation':'WHEN_CORPUS_ACTIVE','termination':'SUPERSEDED_OR_REPEALED_BY_RATIFIED_CORPUS'},'relationships':{'supersedes':[],'superseded_by':[]},'origin':{'source_type':'HUMAN_NOMINATED','source_ref':'candidate'},'notes':[],'history':[]}
@pytest.mark.parametrize('path,value',[('statement',''),('origin.source_type',1),('origin.source_ref',''),('notes',[1]),('history',[{}])])
def test_closed_document_rejects_invalid_leaves(path,value):
 d=doc(); target=d
 bits=path.split('.')
 for bit in bits[:-1]: target=target[bit]
 target[bits[-1]]=value
 with pytest.raises(CanonicalizationError): validate_document(d)
def test_unknown_nested_rejected():
 d=doc(); d['scope']['unknown']='x'
 with pytest.raises(CanonicalizationError): validate_document(d)
