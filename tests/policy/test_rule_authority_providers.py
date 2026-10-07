"""Proveedores confiables del kernel (F1.1 §9, paso 7): interfaces y dobles.

Lo que estas pruebas fijan:
  - versiones monotonicas SIN ABA (un valor repetido no se puede construir);
  - leases compartidos/exclusivos con version estable mientras se sostienen;
  - checkpoint externo monotónico: retroceso y head conflictivo niegan;
  - STOP desconocido, reloj ingenuo o retrocedido, clasificación ausente:
    errores TIPADOS que el kernel traduce en DENY, nunca en permiso;
  - un proveedor que solo relee valores NO satisface el contrato: el guard de
    emisión lo niega — sin leases no se emite ni se consume (§9).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from policy.rule_authority.errors import (
    CheckpointInvalido,
    ClasificacionDesconocida,
    ProveedorInvalido,
    RelojRetrocedio,
    RuleAuthorityError,
    StopDesconocido,
    VersionRetrocedio,
)
from policy.rule_authority.providers import (
    ORDEN_ADQUISICION,
    CheckpointsEnMemoria,
    ClasificacionFija,
    ContratoCapability,
    EstadoStop,
    PinFijo,
    ProveedorSinLeases,
    RelojDeVersiones,
    RelojDeterminista,
    StopFijo,
    StopIlegible,
    VersionMonotonica,
    exigir_contrato_de_emision,
)
from policy.rule_authority.snapshot import TrustedPolicyPin

PIN = TrustedPolicyPin("jax", "0" * 40, "1" * 40, "prueba:paso7")


# ------------------------------------------------------------- versiones sin ABA

def test_version_estrictamente_creciente_ok() -> None:
    v1 = VersionMonotonica(1)
    v2 = VersionMonotonica(2, _previa=v1)
    assert v2.es_posterior_a(v1) and not v1.es_posterior_a(v2)


@pytest.mark.parametrize("numero", [1, 5, 0])       # igual o hacia atras: sin ABA
def test_version_repetida_o_retrocedida_niega(numero: int) -> None:
    v1 = VersionMonotonica(5)
    with pytest.raises(VersionRetrocedio):
        VersionMonotonica(numero, _previa=v1)


def test_reloj_de_versiones_solo_crece() -> None:
    reloj = RelojDeVersiones()
    a, b, c = reloj.siguiente(), reloj.siguiente(), reloj.siguiente()
    assert b.es_posterior_a(a) and c.es_posterior_a(b)


# ------------------------------------------------------------------------ pin

def test_pin_fijo_entrega_el_pin_y_su_procedencia() -> None:
    proveedor = PinFijo(PIN, "checkpoint:externo")
    pin, procedencia = proveedor.pin_activo()
    assert pin == PIN and procedencia == "checkpoint:externo"


def test_el_lease_compartido_sostiene_la_version() -> None:
    proveedor = PinFijo(PIN, "p")
    with proveedor.lease_compartido() as lease:
        sostenida = lease.version_estable
        assert proveedor.version_actual().es_posterior_a(sostenida) or \
            proveedor.version_actual() == sostenida
        # mientras el lease vive, la version estable no cambia
        assert lease.version_estable == sostenida


def test_lease_exclusivo_exige_no_haber_compartidos() -> None:
    proveedor = PinFijo(PIN, "p")
    with proveedor.lease_compartido():
        with pytest.raises(RuleAuthorityError):
            with proveedor.lease_exclusivo():
                pass


# ------------------------------------------------------- checkpoint monotónico

def test_checkpoint_publica_y_entrega_head() -> None:
    checkpoints = CheckpointsEnMemoria()
    checkpoints.publicar("h1", anterior="")
    assert checkpoints.head_actual() == "h1"


def test_checkpoint_rechaza_retroceso() -> None:
    checkpoints = CheckpointsEnMemoria()
    checkpoints.publicar("h1", anterior="")
    checkpoints.publicar("h2", anterior="h1")
    with pytest.raises(CheckpointInvalido):
        checkpoints.publicar("h1", anterior="h2")     # republicar un head ya superado


def test_checkpoint_rechaza_head_conflictivo() -> None:
    checkpoints = CheckpointsEnMemoria()
    checkpoints.publicar("h1", anterior="")
    with pytest.raises(CheckpointInvalido):
        checkpoints.publicar("h3", anterior="otro")   # no encadena con el head real


def test_republicar_el_mismo_head_es_idempotente() -> None:
    checkpoints = CheckpointsEnMemoria()
    checkpoints.publicar("h1", anterior="")
    checkpoints.publicar("h1", anterior="")           # misma foto: ok
    assert checkpoints.head_actual() == "h1"


# ------------------------------------------------------------------------- STOP

def test_stop_conocido_e_inactivo() -> None:
    proveedor = StopFijo(activo=False, huella="sha256:x")
    estado = proveedor.estado()
    assert isinstance(estado, EstadoStop) and estado.activo is False


def test_stop_desconocido_es_error_tipado() -> None:
    with pytest.raises(StopDesconocido):
        StopIlegible().estado()


# ------------------------------------------------------------------------ reloj

def test_reloj_entrega_utc_consciente() -> None:
    momento = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)
    reloj = RelojDeterminista(momento)
    assert reloj.ahora() == momento


def test_reloj_ingenuo_niega() -> None:
    reloj = RelojDeterminista(datetime(2026, 10, 6, 12, 0, 0))    # sin tzinfo
    with pytest.raises(RuleAuthorityError):
        reloj.ahora()


def test_reloj_detecta_retroceso_respecto_de_emision() -> None:
    reloj = RelojDeterminista(datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc))
    reloj.marcar_emision(reloj.ahora())
    reloj.fijar(datetime(2026, 10, 6, 11, 59, 59, tzinfo=timezone.utc))
    with pytest.raises(RelojRetrocedio):
        reloj.ahora()


# --------------------------------------------------------------- clasificación

def test_clasificacion_conocida_entrega_contrato() -> None:
    contrato = ContratoCapability(identidad="cap-x", version="v1", clase="OBLIGATING",
                                  unidad="mensajes", moneda=None)
    proveedor = ClasificacionFija({"CAP_X": contrato})
    assert proveedor.contrato_de("CAP_X") == contrato


def test_clasificacion_ausente_es_error_tipado() -> None:
    proveedor = ClasificacionFija({})
    with pytest.raises(ClasificacionDesconocida):
        proveedor.contrato_de("CAP_DESCONOCIDA")


# --------------------------------------------------- el guard de emisión (§9)

def _suite_completa() -> dict:
    return {
        "pin": PinFijo(PIN, "p"),
        "checkpoint": CheckpointsEnMemoria(),
        "stop": StopFijo(activo=False, huella="x"),
        "reloj": RelojDeterminista(datetime(2026, 10, 6, tzinfo=timezone.utc)),
        "clasificacion": ClasificacionFija({}),
    }


def test_el_guard_exige_todos_los_proveedores() -> None:
    exigir_contrato_de_emision(**_suite_completa())         # no lanza
    for ausente in ("pin", "checkpoint", "stop", "reloj", "clasificacion"):
        suite = _suite_completa()
        suite[ausente] = None
        with pytest.raises(ProveedorInvalido):
            exigir_contrato_de_emision(**suite)


def test_proveedor_que_solo_relee_no_puede_emitir_ni_consumir() -> None:
    for campo in ("pin", "stop", "clasificacion"):
        suite = _suite_completa()
        suite[campo] = ProveedorSinLeases()                # relee valores, sin leases
        with pytest.raises(ProveedorInvalido) as excinfo:
            exigir_contrato_de_emision(**suite)
        assert "lease" in str(excinfo.value)


def test_el_orden_global_de_adquisicion_es_el_del_diseno() -> None:
    assert ORDEN_ADQUISICION == ("pin_activo", "clasificacion", "stop", "permit_db",
                                 "authority_ledger_head", "audit_head")
