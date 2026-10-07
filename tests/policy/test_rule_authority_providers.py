"""Proveedores confiables (F1.1 paso 7 r2): interfaces, guard y dobles reales.

Portados los ataques A1-A14 del auditor como regresiones, con carreras REALES
de hilos (0 lecturas durante exclusivo; 0 bifurcaciones de checkpoint) y los
mutantes G/M del guion reconstruidos contra el diseño nuevo.
"""
from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from proveedores_dobles import (  # noqa: E402
    CheckpointsEnMemoria,
    ClasificacionFija,
    PinFijo,
    ProveedorSinLeases,
    RelojDeterminista,
    StopFijo,
    StopIlegible,
)

from policy.rule_authority.errors import (
    CheckpointInvalido,
    ClasificacionDesconocida,
    ProveedorInvalido,
    RelojInvalido,
    RelojRetrocedio,
    RuleAuthorityError,
    StopDesconocido,
)
from policy.rule_authority.providers import (
    ORDEN_ADQUISICION,
    ClaseCapability,
    ContratoCapability,
    EstadoStop,
    VersionMonotonica,
    VistaLease,
    exigir_contrato_de_emision,
)
from policy.rule_authority.snapshot import TrustedPolicyPin

PIN = TrustedPolicyPin("jax", "0" * 40, "1" * 40, "prueba:paso7r2")
T0 = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)


def _suite(checkpoint_publicado: bool = True) -> dict:
    checkpoint = CheckpointsEnMemoria()
    if checkpoint_publicado:
        checkpoint.publicar("h1", anterior="")
    return {
        "pin": PinFijo(PIN, "checkpoint:externo"),
        "checkpoint": checkpoint,
        "stop": StopFijo(activo=False, huella="sha256:x"),
        "reloj": RelojDeterminista(T0),
        "clasificacion": ClasificacionFija({}),
    }


# ------------------------------------------------------------ tipos de valor

def test_a1_version_valida_entero_no_negativo_y_compara() -> None:
    assert VersionMonotonica(5).numero == 5
    assert VersionMonotonica(0).numero == 0
    assert VersionMonotonica(5) == VersionMonotonica(5)      # forma; el ORDEN es del emisor
    assert VersionMonotonica(6).es_posterior_a(VersionMonotonica(5))


@pytest.mark.parametrize("malo", [True, -1, "5", 1.5, None])
def test_version_invalida_niega(malo: object) -> None:
    with pytest.raises(ProveedorInvalido):
        VersionMonotonica(malo)                               # type: ignore[arg-type]


def test_version_inmutable() -> None:
    v = VersionMonotonica(1)
    with pytest.raises(RuleAuthorityError):
        v.numero = 2                                          # type: ignore[misc]


def test_m10_version_negativa_niega() -> None:
    with pytest.raises(ProveedorInvalido):
        VersionMonotonica(-1)


def test_m2_el_compartido_libera_de_verdad() -> None:
    """M2 (fuga de lectores): si el compartido no decrementara, el exclusivo
    quedaria esperando para siempre. Con hilo y plazo: entra o queda claro."""
    proveedor = PinFijo(PIN, "p")
    with proveedor.lease_compartido():
        pass                                                # al salir: cero lectores
    entro: list[bool] = []

    def escritor() -> None:
        with proveedor.lease_exclusivo():
            entro.append(True)

    hilo = threading.Thread(target=escritor, daemon=True)
    hilo.start()
    hilo.join(timeout=2.0)
    assert entro, "el compartido no libero: el exclusivo no entra nunca (fuga)"


# ------------------------------------------------------------------- leases

def test_a2_el_exclusivo_excluye_de_verdad() -> None:
    proveedor = PinFijo(PIN, "p")
    with proveedor.lease_exclusivo():
        with pytest.raises(Exception):                        # un compartido NUNCA entra
            with proveedor.lease_exclusivo():
                pass


def test_el_lease_entrega_valor_y_version_del_mismo_instante() -> None:
    proveedor = PinFijo(PIN, "p")
    with proveedor.lease_compartido() as vista:
        assert isinstance(vista, VistaLease)
        assert vista.valor[0] == PIN and isinstance(vista.version, VersionMonotonica)


def test_m9_la_version_es_estable_mientras_se_lee() -> None:
    """Si cada lease avanzara la version, esta igualdad estricta rompe. El
    exclusivo entra en hilo con plazo: con una fuga de lectores (M2) queda
    ROJA, no colgada."""
    proveedor = StopFijo(activo=False, huella="x")
    with proveedor.lease_compartido() as v1:
        with proveedor.lease_compartido() as v2:
            assert v1.version == v2.version                   # estricta: sin avance por leer
    avanzo: list[bool] = []

    def escritor() -> None:
        with proveedor.lease_exclusivo() as v3:
            avanzo.append(v3.version.es_posterior_a(v1.version))   # el escritor SI avanza

    hilo = threading.Thread(target=escritor, daemon=True)
    hilo.start()
    hilo.join(timeout=2.0)
    assert avanzo == [True], "el exclusivo no entro (fuga de lectores) o no avanzo"


def test_a3_el_estado_ve_la_version_del_momento() -> None:
    proveedor = StopFijo(activo=False, huella="x")
    antes = proveedor.estado().version
    with proveedor.lease_exclusivo():
        pass
    assert proveedor.estado().version.es_posterior_a(antes)


def test_a14_carrera_real_ninguna_lectura_durante_exclusivo() -> None:
    sys.setswitchinterval(1e-6)
    proveedor = PinFijo(PIN, "p")
    violaciones = [0]
    exclusivo_vivo = [False]
    parar = [False]

    def lector() -> None:
        while not parar[0]:
            with proveedor.lease_compartido():
                if exclusivo_vivo[0]:
                    violaciones[0] += 1

    def escritor() -> None:
        while not parar[0]:
            with proveedor.lease_exclusivo():
                exclusivo_vivo[0] = True
                time.sleep(0)
                exclusivo_vivo[0] = False

    hilos = [threading.Thread(target=lector, daemon=True) for _ in range(3)] + \
            [threading.Thread(target=escritor, daemon=True) for _ in range(2)]
    for h in hilos:
        h.start()
    time.sleep(0.35)
    parar[0] = True
    for h in hilos:
        h.join(timeout=5.0)
        assert not h.is_alive(), "un hilo de la carrera no termino (¿lock que no libera?)"
    assert violaciones[0] == 0, f"{violaciones[0]} lecturas durante exclusivo"


# ---------------------------------------------------------------------- pin

def test_a11_procedencia_sellada_nunca_movil_ni_vacia() -> None:
    with pytest.raises(ProveedorInvalido):
        PinFijo(PIN, "refs/heads/main")
    with pytest.raises(ProveedorInvalido):
        PinFijo(PIN, "")
    proveedor = PinFijo(PIN, "checkpoint:externo")
    assert proveedor.pin_activo()[1] == "checkpoint:externo"


def test_a12_no_hay_lease_fabricable_en_produccion() -> None:
    """El lease solo existe como vista entregada por el proveedor; fabricar una
    VistaLease a mano no da acceso a nada (el kernel consume el context manager)."""
    import policy.rule_authority.providers as P
    assert not hasattr(P, "_Lease")
    falsa = VistaLease(valor=None, version=VersionMonotonica(999))
    assert falsa.valor is None                               # un dato, no una llave


# --------------------------------------------------------------------- stop

def test_a6_estado_stop_validado() -> None:
    with pytest.raises(ProveedorInvalido):
        StopFijo(activo=None, huella="x")                     # type: ignore[arg-type]
    with pytest.raises(ProveedorInvalido):
        StopFijo(activo=False, huella="")
    estado = StopFijo(activo=False, huella="x").estado()
    assert estado.activo is False                             # bool real, no truthy


def test_stop_ilegible_es_tipado() -> None:
    with pytest.raises(StopDesconocido):
        StopIlegible().estado()


# --------------------------------------------------------------------- reloj

def test_m4_reloj_ingenuo_niega() -> None:
    reloj = RelojDeterminista(datetime(2026, 10, 6, 12, 0, 0))   # sin tz
    with pytest.raises(RelojInvalido):
        reloj.ahora()


def test_a7_reloj_monotonico_por_proceso_y_emision_no_baja() -> None:
    reloj = RelojDeterminista(T0)
    reloj.fijar(T0 + timedelta(seconds=10))
    assert reloj.ahora() == T0 + timedelta(seconds=10)
    reloj.fijar(T0 + timedelta(seconds=5))                     # retroceso respecto de la ULTIMA LECTURA
    with pytest.raises(RelojRetrocedio):
        reloj.ahora()
    with pytest.raises(RelojRetrocedio):                       # marcar_emision no baja el piso
        reloj.marcar_emision(T0 - timedelta(days=1))
    reloj.fijar(T0 - timedelta(hours=1))
    with pytest.raises(RelojRetrocedio):
        reloj.ahora()


def test_marcar_emision_sube_el_piso() -> None:
    reloj = RelojDeterminista(T0)
    reloj.marcar_emision(T0 + timedelta(minutes=5))
    reloj.fijar(T0 + timedelta(minutes=1))
    with pytest.raises(RelojRetrocedio):
        reloj.ahora()


# ------------------------------------------------------------- clasificación

def _contrato(clase: ClaseCapability) -> ContratoCapability:
    return ContratoCapability(identidad="cap-x", version="v1", clase=clase,
                              unidad="mensajes", moneda=None)


def test_a8_la_clase_es_enum_cerrado() -> None:
    with pytest.raises(ProveedorInvalido):
        ContratoCapability(identidad="c", version="v", clase="reversible?",   # type: ignore[arg-type]
                           unidad=None, moneda=None)
    with pytest.raises(ProveedorInvalido):
        ClasificacionFija({"D": None})                         # type: ignore[dict-item]


def test_m11_la_ausente_obliga_nunca_rebaja() -> None:
    with pytest.raises(ClasificacionDesconocida):
        ClasificacionFija({}).contrato_de("CAP_X")


def test_sin_rebaja_entre_versiones_del_contrato() -> None:
    proveedor = ClasificacionFija({"CAP_X": _contrato(ClaseCapability.OBLIGATING)})
    with pytest.raises(ProveedorInvalido):
        proveedor.republicar("CAP_X", _contrato(ClaseCapability.REVERSIBLE))
    proveedor.republicar("CAP_X", _contrato(ClaseCapability.OBLIGATING))   # igual: ok
    proveedor.republicar("CAP_X", _contrato(ClaseCapability.OBLIGATING))   # subir: ok


# ---------------------------------------------------------------- checkpoint

def test_a9_idempotencia_solo_con_el_mismo_anterior() -> None:
    checkpoints = CheckpointsEnMemoria()
    checkpoints.publicar("h1", anterior="")
    checkpoints.publicar("h2", anterior="h1")
    with pytest.raises(CheckpointInvalido):
        checkpoints.publicar("h2", anterior="")                # head repetido, anterior DISTINTO
    with pytest.raises(CheckpointInvalido):
        checkpoints.publicar("h2", anterior="h2")
    checkpoints.publicar("h2", anterior="h1")                  # idempotencia exacta
    assert checkpoints.head_actual() == "h2"


def test_g1_retroceso_y_conflicto_niegan() -> None:
    checkpoints = CheckpointsEnMemoria()
    checkpoints.publicar("h1", anterior="")
    checkpoints.publicar("h2", anterior="h1")
    with pytest.raises(CheckpointInvalido):
        checkpoints.publicar("h1", anterior="h2")
    with pytest.raises(CheckpointInvalido):
        checkpoints.publicar("h3", anterior="otro")


def test_confirmar_relee_el_log() -> None:
    checkpoints = CheckpointsEnMemoria()
    checkpoints.publicar("h1", anterior="")
    assert checkpoints.confirmar("h1") is True
    assert checkpoints.confirmar("h9") is False


def test_a13_carrera_hilos_cero_bifurcaciones() -> None:
    sys.setswitchinterval(1e-6)
    bifurcaciones = 0
    for _ in range(300):
        checkpoints = CheckpointsEnMemoria()
        checkpoints.publicar("h0", anterior="")
        exitosos: list[str] = []
        barrera = threading.Barrier(2)

        def escritor(head: str) -> None:
            barrera.wait()
            try:
                checkpoints.publicar(head, anterior="h0")
                exitosos.append(head)
            except CheckpointInvalido:
                pass

        hilos = [threading.Thread(target=escritor, args=(h,)) for h in ("hA", "hB")]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()
        if len(exitosos) == 2:
            bifurcaciones += 1
    assert bifurcaciones == 0, f"{bifurcaciones} bifurcaciones aceptadas"


# -------------------------------------------------------------------- guard

def test_a4_el_guard_niega_mocks() -> None:
    with pytest.raises(ProveedorInvalido):
        exigir_contrato_de_emision(pin=Mock(), checkpoint=Mock(), stop=Mock(),
                                    reloj=Mock(), clasificacion=Mock())


class _Falso:
    def pin_activo(self):
        return None

    def estado(self):
        return EstadoStop(activo=None, version=None, huella="")   # type: ignore[arg-type]

    def contrato_de(self, capability: str):
        return None

    def lease_compartido(self):
        return None

    def lease_exclusivo(self):
        return None


def test_a5_el_guard_niega_vistas_y_valores_none() -> None:
    suite = _suite()
    for campo in ("pin", "stop", "clasificacion"):
        mala = _suite()
        mala[campo] = _Falso()
        with pytest.raises(ProveedorInvalido):
            exigir_contrato_de_emision(**mala)
    assert suite  # silencia linters; el bucle ya probo los tres


def test_a10_checkpoint_nunca_publicado_niega() -> None:
    suite = _suite(checkpoint_publicado=False)
    with pytest.raises(ProveedorInvalido) as excinfo:
        exigir_contrato_de_emision(**suite)
    assert "nunca publicado" in str(excinfo.value)


def test_el_guard_pasa_la_suite_completa() -> None:
    exigir_contrato_de_emision(**_suite())                    # no lanza


class _PinConLeasesRotos:
    """Pasa el isinstance (tiene todos los metodos) y entrega un pin VALIDO;
    solo sus leases estan rotos (devuelven None). Si el guard no mirara la
    vista del lease, este doble pasaria."""

    def lease_compartido(self):
        return None

    def lease_exclusivo(self):
        return None

    def pin_activo(self):
        return (PIN, "checkpoint:externo")


def test_g2_m12_el_guard_exige_leases_con_vista_valida() -> None:
    for campo, roto in (("pin", _PinConLeasesRotos()), ):
        suite = _suite()
        suite[campo] = roto
        with pytest.raises(ProveedorInvalido) as excinfo:
            exigir_contrato_de_emision(**suite)
        assert "lease" in str(excinfo.value) or "VistaLease" in str(excinfo.value)
    for campo in ("pin", "stop", "clasificacion"):
        suite = _suite()
        suite[campo] = ProveedorSinLeases()          # ni siquiera implementa el Protocolo
        with pytest.raises(ProveedorInvalido):
            exigir_contrato_de_emision(**suite)


def test_falta_cualquier_proveedor_niega() -> None:
    for campo in ("pin", "checkpoint", "stop", "reloj", "clasificacion"):
        suite = _suite()
        suite[campo] = None
        with pytest.raises(ProveedorInvalido):
            exigir_contrato_de_emision(**suite)


def test_el_orden_global_de_adquisicion_es_el_del_diseno() -> None:
    assert ORDEN_ADQUISICION == ("pin_activo", "clasificacion", "stop", "permit_db",
                                 "authority_ledger_head", "audit_head")


# --------------------------------- mutantes M6/M7/M8/M12: la razón propia

def test_m8_versiones_iguales_no_son_posteriores() -> None:
    assert VersionMonotonica(5).es_posterior_a(VersionMonotonica(5)) is False


class _PinSinExclusivo:
    """lease_compartido FUNCIONA; lease_exclusivo existe como atributo pero no
    es llamable (pasa el isinstance de runtime_checkable, que solo mira
    atributos). Si el guard no lo exigiera llamable, este doble pasaria."""

    def __init__(self) -> None:
        self._base = _BasePrestada()

    def lease_compartido(self):
        return self._base.lease_compartido()

    lease_exclusivo = None

    def pin_activo(self):
        return (PIN, "checkpoint:externo")


class _BasePrestada:
    def __init__(self) -> None:
        from proveedores_dobles import PinFijo
        self._real = PinFijo(PIN, "p")

    def lease_compartido(self):
        return self._real.lease_compartido()


def test_m12_el_guard_exige_lease_exclusivo_llamable() -> None:
    suite = _suite()
    suite["pin"] = _PinSinExclusivo()
    with pytest.raises(ProveedorInvalido):
        exigir_contrato_de_emision(**suite)
    # La capa que mata a M12 es el isinstance runtime_checkable de 3.12+
    # (atributos no llamables no implementan el Protocolo); el chequeo explicito
    # del guard es cinturon y tirantes.


class _CheckpointAtributosMuertos:
    """head_actual funciona; publicar/confirmar existen pero no llaman."""

    def __init__(self) -> None:
        self._real = CheckpointsEnMemoria()
        self._real.publicar("h1", anterior="")

    def head_actual(self) -> str:
        return self._real.head_actual()

    publicar = None
    confirmar = None


def test_m6_el_guard_exige_publicar_y_confirmar_llamables() -> None:
    suite = _suite()
    suite["checkpoint"] = _CheckpointAtributosMuertos()
    with pytest.raises(ProveedorInvalido):
        exigir_contrato_de_emision(**suite)
    # M6 muere por el isinstance (callable en 3.12+); ver nota de M12.


class _RelojSinMarcar:
    def __init__(self) -> None:
        self._real = RelojDeterminista(T0)

    def ahora(self) -> object:
        return self._real.ahora()

    marcar_emision = None


def test_m7_el_guard_exige_marcar_emision_llamable() -> None:
    suite = _suite()
    suite["reloj"] = _RelojSinMarcar()
    with pytest.raises(ProveedorInvalido):
        exigir_contrato_de_emision(**suite)
    # M7 muere por el isinstance (callable en 3.12+); ver nota de M12.


class _PinConVistaTrucha:
    """El lease FUNCIONA (context manager de verdad) pero la vista trae una
    version que no es VersionMonotonica. Sin la validacion de la vista, pasa."""

    def lease_compartido(self):
        from contextlib import contextmanager
        @contextmanager
        def cm():
            yield VistaLease(valor=(PIN, "checkpoint:externo"), version=99)
        return cm()

    def lease_exclusivo(self):
        return self.lease_compartido()

    def pin_activo(self):
        return (PIN, "checkpoint:externo")


def test_m5_la_vista_del_lease_se_valida() -> None:
    suite = _suite()
    suite["pin"] = _PinConVistaTrucha()
    with pytest.raises(ProveedorInvalido) as excinfo:
        exigir_contrato_de_emision(**suite)
    assert "VistaLease" in str(excinfo.value) or "vista" in str(excinfo.value)
