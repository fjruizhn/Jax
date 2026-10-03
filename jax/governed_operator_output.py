"""Closed F2-E operator text boundary for external alerts."""
from __future__ import annotations
import hashlib, json, uuid
from datetime import datetime, timezone
from policy.governance.runtime_output_composition import RuntimeOutputComposition
from policy.governance.runtime_status import build_owned_runtime_status_registry
from policy.governance.governed_renderer import GovernedRenderer
from policy.governance.output_lifecycle import mint_governed_transport_unit
from policy.governance.operator_transport import OperatorTransportMetadata, mint_operator_wire, revalidate_operator_wire
from policy.governance.response import ResponseScope
from jax.external_output import runtime_status_templates

class GovernedOperatorOutputAdapter:
    def __init__(self):
        self._composer=RuntimeOutputComposition(platform_source_configuration=None, registry_factory=lambda s,a: build_owned_runtime_status_registry(s, authenticator=a), governance_receipt=None, templates=runtime_status_templates())
    async def telegram(self, *, message: str, chat_id: str):
        if not isinstance(message,str) or not isinstance(chat_id,str) or not chat_id: raise ValueError('operator alert invalid')
        scope=ResponseScope('production','system',None,None,'service:jax','operator','jacobs.telegram-alert','operator:'+str(uuid.uuid4()),'operator:'+str(uuid.uuid4()))
        composed=await self._composer.compose(scope=scope,response_id='operator:'+str(uuid.uuid4()),producer='jacobs.telegram-alert',tool_data={'message':message},requests=(),number_binding_contract_id=None)
        rendered=GovernedRenderer().render_text(composed.envelope,composed.context)
        unit=mint_governed_transport_unit(composed.envelope,rendered,composed.context,transport_kind='external-alert',idempotency_key='telegram:'+str(uuid.uuid4()))
        identity='sha256:'+hashlib.sha256(chat_id.encode()).hexdigest()
        return mint_operator_wire(unit,rendered,OperatorTransportMetadata('EXTERNAL_ALERT_V1','telegram-json-v1',identity,chat_id=chat_id),now=datetime.now(timezone.utc))

_ADAPTER=None
def operator_adapter():
 global _ADAPTER
 if _ADAPTER is None:_ADAPTER=GovernedOperatorOutputAdapter()
 return _ADAPTER
