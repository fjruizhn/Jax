"""Helper de PRUEBA: estampa el sello Faro individual sobre un grant de regla.

No es código de producción. El sello (`init=False`) lo estampa quien ya tiene
un snapshot Faro validado; en producción esa vía aún no existe, así que las
pruebas lo estampan aquí de forma explícita y visible, accediendo al
centinela privado del módulo como lo haría cualquier prueba de contrato.
"""
from policy.authority_ledger import models
from policy.authority_ledger.models import (
    AuthorityEventIntent, AuthorityEventType, RuleRatificationGrantPayload,
)


def rule_grant_intent(rule_ratification: RuleRatificationGrantPayload,
                      evidence_refs: tuple[str, ...] = ()) -> AuthorityEventIntent:
    intent = AuthorityEventIntent(
        AuthorityEventType.RULE_RATIFICATION_GRANTED, "human:fernando", evidence_refs,
        rule_ratification=rule_ratification,
    )
    object.__setattr__(intent, "_rule_ratification_snapshot_seal",
                       models._RULE_RATIFICATION_SNAPSHOT_SEAL)
    return intent
