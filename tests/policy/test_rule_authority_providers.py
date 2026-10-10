"""Proveedores confiables (F1.1 paso 7 r3): interfaces, guard y dobles reales.

Los dobles viven en ``proveedores_dobles.py`` (solo pruebas). Aqui:

  - el guard se prueba con leases VALIDOS y SOLO el dato malo, para que cada
    validacion muera por SU razon (X1-X4), no por el chequeo del lease;
  - la exclusion de los leases se prueba de forma DETERMINISTA (el hilo que
    intenta entrar o entra —y la prueba cae— o queda esperando, observable en
    ``esperando()``): sin dormir a ver si «tiene suerte»;
  - las carreras con hilos que quedan son de estres adicional, no la prueba.
"""
from __future__ import annotations

import ast
import inspect
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from proveedores_dobles import (  # noqa: E402
    CatalogoFijo,
    CheckpointsEnMemoria,
    ClasificacionFija,
    PinFijo,
    ProveedorSinLeases,
    RelojDeterminista,
    StopFijo,
    StopIlegible,
)

from policy.rule_authority.errors import (  # noqa: E402
    CheckpointInvalido,
    ClasificacionDesconocida,
    ProveedorInvalido,
    RelojInvalido,
    RelojRetrocedio,
    RuleAuthorityError,
    StopDesconocido,
)
from policy.rule_authority.providers import (  # noqa: E402
    ORDEN_ADQUISICION,
    ClaseCapability,
    ContratoCapability,
    EstadoStop,
    FormaLimites,
    PinActivo,
    VersionMonotonica,
    VistaLease,
    exigir_contrato_de_emision,
    leases_de_emision,
)
from policy.rule_authority.snapshot import TrustedPolicyPin  # noqa: E402
from policy.authority_ledger.trusted_checkpoint import TrustedCheckpointStore  # noqa: E402

PROC = "refs/heads/main"
PIN = TrustedPolicyPin("jax", "0" * 40, "1" * 40, PROC)
T0 = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)
RAIZ = Path(__file__).resolve().parents[2]


class _SiempreMayor(int):
    def __gt__(self, otro: object) -> bool:
        return True

    def __eq__(self, otro: object) -> bool:
        return False


class _SiempreIgual(str):
    def __eq__(self, otro: object) -> bool:
        return True

    __hash__ = str.__hash__


def _contrato(clase: ClaseCapability) -> ContratoCapability:
    if clase is ClaseCapability.REVERSIBLE:
        return ContratoCapability(identidad="cap-x", version="v1", clase=clase,
                                  unidad=None, moneda=None,
                                  forma_limites=FormaLimites.NINGUNA)
    return ContratoCapability(identidad="cap-x", version="v1", clase=clase,
                              unidad="mensajes", moneda=None,
                              forma_limites=FormaLimites.CANTIDAD)


def _suite(checkpoint_publicado: bool = True, **cambios: object) -> dict:
    checkpoint = CheckpointsEnMemoria()
    if checkpoint_publicado:
        checkpoint.publicar("h1", anterior="")
    base: dict = {
        "pin": PinFijo(PIN, PROC),
        "rule_audit_checkpoint": checkpoint,
        "stop": StopFijo(activo=False, huella="sha256:x"),
        "reloj": RelojDeterminista(T0),
        "clasificacion": ClasificacionFija({"CAP_X": _contrato(ClaseCapability.OBLIGATING)}),
    }
    base.update(cambios)
    return base


def _niega(motivo: str, **cambios: object) -> None:
    """El guard NIEGA con ProveedorInvalido cuyo mensaje contiene ``motivo``: la
    razon importa (un lease valido + dato malo no puede caer por otra cosa)."""
    with pytest.raises(ProveedorInvalido, match=motivo):
        exigir_contrato_de_emision(**_suite(**cambios))


class _Crudo:
    """Proveedor con leases VALIDOS (context manager real) que entrega
    EXACTAMENTE el valor y la version que se le digan, sin validar nada."""

    def __init__(self, valor: object, version: object = None) -> None:
        self._valor = valor
        self._version = VersionMonotonica(1) if version is None else version

    @contextmanager
    def lease_compartido(self):
        yield VistaLease(self._valor, self._version)       # type: ignore[arg-type]

    def lease_exclusivo(self):
        return self.lease_compartido()


class _RelojCrudo:
    """Reloj que entrega TAL CUAL lo que se le diga (los dobles lo normalizan a UTC)."""

    def __init__(self, momento: object) -> None:
        self._momento = momento

    def ahora(self) -> object:
        return self._momento

    def marcar_emision(self, momento: datetime) -> None:
        return None


@contextmanager
def _intervalo_de_conmutacion(valor: float):
    """Cambia el intervalo de conmutacion de hilos y SIEMPRE lo restaura."""
    original = sys.getswitchinterval()
    sys.setswitchinterval(valor)
    try:
        yield
    finally:
        sys.setswitchinterval(original)


@pytest.fixture
def intervalo_corto():
    with _intervalo_de_conmutacion(1e-6):
        yield


def test_el_intervalo_de_conmutacion_se_restaura_aun_si_el_cuerpo_falla() -> None:
    original = sys.getswitchinterval()
    with pytest.raises(RuntimeError):
        with _intervalo_de_conmutacion(1e-6):
            assert sys.getswitchinterval() != original
            raise RuntimeError("falla el cuerpo")
    assert sys.getswitchinterval() == original


# ------------------------------------------------------------ tipos de valor

def test_a1_version_valida_entero_no_negativo_y_compara() -> None:
    assert VersionMonotonica(5).numero == 5
    assert VersionMonotonica(0).numero == 0
    assert VersionMonotonica(5) == VersionMonotonica(5)      # forma; el ORDEN es del emisor
    assert VersionMonotonica(6).es_posterior_a(VersionMonotonica(5))


def test_version_rechaza_subclase_int_con_comparacion_hostil() -> None:
    with pytest.raises(ProveedorInvalido):
        VersionMonotonica(_SiempreMayor(0))


@pytest.mark.parametrize("campo", ["identidad", "version", "unidad", "moneda"])
def test_contrato_rechaza_subclases_str_en_campos_de_identidad(campo: str) -> None:
    datos = {"identidad": "cap-x", "version": "v1", "clase": ClaseCapability.OBLIGATING,
             "unidad": "mensajes", "moneda": "HNL", "forma_limites": FormaLimites.CANTIDAD}
    datos[campo] = _SiempreIgual(str(datos[campo]))
    with pytest.raises(ProveedorInvalido):
        ContratoCapability(**datos)  # type: ignore[arg-type]


@pytest.mark.parametrize("malo", [True, -1, "5", 1.5, None])
def test_version_invalida_niega(malo: object) -> None:
    with pytest.raises(ProveedorInvalido):
        VersionMonotonica(malo)                               # type: ignore[arg-type]


def test_version_inmutable_sin_setter_ni_deleter() -> None:
    v = VersionMonotonica(1)
    with pytest.raises(RuleAuthorityError):
        v.numero = 2                                          # type: ignore[misc]
    with pytest.raises(RuleAuthorityError):
        del v._numero
    assert v.numero == 1


def test_contrato_inmutable_sin_setter_ni_deleter() -> None:
    c = _contrato(ClaseCapability.OBLIGATING)
    with pytest.raises(RuleAuthorityError):
        c.clase = ClaseCapability.REVERSIBLE                  # type: ignore[misc]
    for campo in ("clase", "forma_limites", "identidad"):
        with pytest.raises(RuleAuthorityError):
            delattr(c, campo)
    assert c.clase is ClaseCapability.OBLIGATING and c.forma_limites is FormaLimites.CANTIDAD


def test_forma_de_limites_es_enum_cerrado_y_entra_en_la_identidad() -> None:
    """§10.7: el consumo revalida la forma de limites; un contrato que cambia
    SOLO la forma no es el mismo contrato."""
    with pytest.raises(ProveedorInvalido):
        ContratoCapability(identidad="c", version="v", clase=ClaseCapability.OBLIGATING,
                           unidad=None, moneda=None, forma_limites="cantidad")   # type: ignore[arg-type]
    a = _contrato(ClaseCapability.OBLIGATING)
    b = ContratoCapability(identidad="cap-x", version="v1", clase=ClaseCapability.OBLIGATING,
                           unidad=None, moneda="HNL", forma_limites=FormaLimites.MONTO)
    assert a != b and hash(a) != hash(b)


@pytest.mark.parametrize(
    "clase,unidad,moneda,forma",
    [
        (ClaseCapability.OBLIGATING, None, None, FormaLimites.NINGUNA),
        (ClaseCapability.OBLIGATING, None, "HNL", FormaLimites.CANTIDAD),
        (ClaseCapability.OBLIGATING, "mensajes", None, FormaLimites.MONTO),
        (ClaseCapability.REVERSIBLE, "mensajes", None, FormaLimites.NINGUNA),
        (ClaseCapability.REVERSIBLE, None, "HNL", FormaLimites.NINGUNA),
    ],
)
def test_contrato_rechaza_forma_de_limites_incoherente(
    clase: ClaseCapability, unidad: str | None, moneda: str | None,
    forma: FormaLimites,
) -> None:
    with pytest.raises(ProveedorInvalido):
        ContratoCapability(identidad="cap-x", version="v1", clase=clase,
                           unidad=unidad, moneda=moneda, forma_limites=forma)


def test_forma_limites_solo_contiene_los_tres_valores_del_schema() -> None:
    assert {item.value for item in FormaLimites} == {"NINGUNA", "CANTIDAD", "MONTO"}


def test_reversible_puede_declarar_forma_cantidad_o_monto_coherente() -> None:
    cantidad = ContratoCapability(identidad="cap-c", version="v1",
                                  clase=ClaseCapability.REVERSIBLE,
                                  unidad="mensajes", moneda=None,
                                  forma_limites=FormaLimites.CANTIDAD)
    monto = ContratoCapability(identidad="cap-m", version="v1",
                               clase=ClaseCapability.REVERSIBLE,
                               unidad=None, moneda="HNL",
                               forma_limites=FormaLimites.MONTO)
    assert cantidad.forma_limites is FormaLimites.CANTIDAD
    assert monto.forma_limites is FormaLimites.MONTO


def test_m10_m8_version_negativa_niega_e_iguales_no_son_posteriores() -> None:
    with pytest.raises(ProveedorInvalido):
        VersionMonotonica(-1)
    assert VersionMonotonica(5).es_posterior_a(VersionMonotonica(5)) is False


# ------------------------------------------------------------------- leases

def _intento_en_hilo(abrir):
    """Hilo que intenta ``with abrir():``. Devuelve (entro, soltar, hilo)."""
    entro, soltar = threading.Event(), threading.Event()

    def correr() -> None:
        with abrir():
            entro.set()
            soltar.wait(timeout=5.0)

    hilo = threading.Thread(target=correr, daemon=True)
    hilo.start()
    return entro, soltar, hilo


def _hasta_entrar_o_esperar(proveedor: object, entro: threading.Event) -> None:
    """DETERMINISTA: vuelve cuando el hilo YA ENTRO (la prueba va a caer) o YA
    ESTA ESPERANDO en el lock. Sin dormir un tiempo y rezar."""
    limite = time.monotonic() + 5.0
    while not entro.is_set() and proveedor.esperando() == 0:      # type: ignore[attr-defined]
        assert time.monotonic() < limite, "el hilo ni entro ni espero (¿murio?)"
        time.sleep(0.0005)


@pytest.mark.parametrize("duenyo,intento,razon", [
    ("lease_exclusivo", "lease_compartido", "M1: el compartido entro con un exclusivo vivo"),
    ("lease_exclusivo", "lease_exclusivo", "X7: dos exclusivos a la vez (los escritores no se excluyen)"),
    ("lease_compartido", "lease_exclusivo", "X9: el exclusivo entro con un compartido vivo (no espera)"),
])
def test_a2_la_exclusion_es_real_de_hilo_a_hilo(duenyo: str, intento: str, razon: str) -> None:
    """El hilo que intenta entrar DEBE quedar esperando mientras el dueño vive y
    entrar apenas lo suelta. Hilos distintos: la reentrada del mismo hilo (que el
    doble niega aparte) no puede hacer pasar esta prueba por la razon equivocada."""
    proveedor = PinFijo(PIN, PROC)
    soltar = None
    try:
        with getattr(proveedor, duenyo)():
            entro, soltar, hilo = _intento_en_hilo(getattr(proveedor, intento))
            _hasta_entrar_o_esperar(proveedor, entro)
            assert not entro.is_set(), razon
        assert entro.wait(timeout=5.0), "soltado el dueño, el que esperaba no entro (¿lock que no libera?)"
    finally:
        if soltar is not None:
            soltar.set()
    hilo.join(timeout=5.0)
    assert not hilo.is_alive()


def test_un_lector_nuevo_no_se_adelanta_a_un_escritor_que_ya_espera() -> None:
    proveedor = PinFijo(PIN, PROC)
    escritor_entro, soltar_escritor = threading.Event(), threading.Event()
    lector_entro, soltar_lector = threading.Event(), threading.Event()

    def escritor() -> None:
        with proveedor.lease_exclusivo():
            escritor_entro.set()
            soltar_escritor.wait(timeout=5.0)

    def lector() -> None:
        with proveedor.lease_compartido():
            lector_entro.set()
            soltar_lector.wait(timeout=5.0)

    with proveedor.lease_compartido():
        hilo_escritor = threading.Thread(target=escritor, daemon=True)
        hilo_escritor.start()
        limite = time.monotonic() + 5.0
        while proveedor.esperando() < 1:
            assert time.monotonic() < limite, "el escritor no quedo esperando"
            time.sleep(0.0005)
        hilo_lector = threading.Thread(target=lector, daemon=True)
        hilo_lector.start()
        while proveedor.esperando() < 2 and not lector_entro.is_set():
            assert time.monotonic() < limite, "el lector no quedo esperando"
            time.sleep(0.0005)
        assert not escritor_entro.is_set() and not lector_entro.is_set()

    try:
        assert escritor_entro.wait(timeout=5.0), "el escritor no progreso al liberar el lector"
        assert not lector_entro.is_set(), "un lector nuevo adelanto al escritor pendiente"
    finally:
        soltar_escritor.set()
    assert lector_entro.wait(timeout=5.0), "el lector no progreso despues del escritor"
    soltar_lector.set()
    hilo_escritor.join(timeout=5.0)
    hilo_lector.join(timeout=5.0)
    assert not hilo_escritor.is_alive() and not hilo_lector.is_alive()


def test_dos_compartidos_coexisten() -> None:
    """Si los lectores se excluyeran entre si, la barrera expira (BrokenBarrierError)."""
    proveedor = PinFijo(PIN, PROC)
    barrera = threading.Barrier(2, timeout=5.0)
    errores: list[BaseException] = []

    def lector() -> None:
        try:
            with proveedor.lease_compartido():
                barrera.wait()                     # los DOS dentro a la vez
        except threading.BrokenBarrierError as exc:
            errores.append(exc)

    hilos = [threading.Thread(target=lector, daemon=True) for _ in range(2)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout=10.0)
    assert errores == []


@pytest.mark.parametrize("externo,interno", [
    ("lease_exclusivo", "lease_compartido"),      # compartido DENTRO de exclusivo
    ("lease_exclusivo", "lease_exclusivo"),
    ("lease_compartido", "lease_exclusivo"),      # promocion
])
def test_reentrada_del_mismo_hilo_es_error_tipado_no_bloqueo(externo: str, interno: str) -> None:
    """Declarado en el Protocolo: reentrada y promocion del mismo hilo NO se
    soportan y fallan al instante (sin deadlock)."""
    proveedor = PinFijo(PIN, PROC)
    with getattr(proveedor, externo)():
        with pytest.raises(RuleAuthorityError, match="no soportada"):
            with getattr(proveedor, interno)():
                pass
    with proveedor.lease_exclusivo():              # y el lock quedo sano
        pass


def test_el_lease_entrega_valor_y_version_del_mismo_instante_con_la_misma_forma() -> None:
    proveedor = PinFijo(PIN, PROC)
    with proveedor.lease_compartido() as compartida:
        pass
    with proveedor.lease_exclusivo() as exclusiva:
        pass
    for vista in (compartida, exclusiva):
        assert isinstance(vista, VistaLease) and isinstance(vista.version, VersionMonotonica)
        assert type(vista.valor) is PinActivo and vista.valor.pin == PIN
    assert type(compartida.valor) is type(exclusiva.valor)


def test_m2_el_compartido_libera_de_verdad() -> None:
    """Fuga de lectores: si el compartido no decrementara, el exclusivo esperaria
    para siempre. Hilo con plazo: cae roja, no cuelga."""
    proveedor = PinFijo(PIN, PROC)
    with proveedor.lease_compartido():
        pass
    entro, soltar, hilo = _intento_en_hilo(proveedor.lease_exclusivo)
    try:
        assert entro.wait(timeout=5.0), "el compartido no libero: el exclusivo no entra nunca"
    finally:
        soltar.set()
    hilo.join(timeout=5.0)


def test_m9_la_version_es_estable_mientras_se_lee() -> None:
    """Si cada lease avanzara la version, esta igualdad estricta rompe."""
    proveedor = StopFijo(activo=False, huella="x")
    with proveedor.lease_compartido() as v1:
        with proveedor.lease_compartido() as v2:
            assert v1.version == v2.version                   # estricta: sin avance por leer
    with proveedor.lease_exclusivo() as v3:
        assert v3.version == v1.version                        # ENTRAR no avanza la version
        proveedor._reemplazar(EstadoStop(activo=True, huella="y"))   # escribir: valor y version juntos
    with proveedor.lease_compartido() as v4:
        assert v4.version.es_posterior_a(v1.version)
        assert v4.valor == EstadoStop(activo=True, huella="y")  # el compartido posterior ve lo escrito
    assert v3.valor == EstadoStop(activo=False, huella="x")     # la vista del exclusivo: el instante de entrada
    with pytest.raises(RuleAuthorityError, match="exclusivo"):  # y no se escribe sin el exclusivo
        proveedor._reemplazar(EstadoStop(activo=False, huella="z"))


def test_los_leases_de_emision_mantienen_las_tres_vistas_hasta_el_fin_del_contexto() -> None:
    """Rompe si la frontera cierra algun lease antes de que quien decide termine.

    El cambio que debe volverla roja es adquirir y validar cada lease por
    separado: entonces el escritor de ese proveedor entraria dentro del
    contexto, antes de que la persistencia de una decision pueda terminar.
    """
    pin = PinFijo(PIN, PROC)
    clasificacion = ClasificacionFija({"CAP_X": _contrato(ClaseCapability.OBLIGATING)})
    stop = StopFijo(activo=False, huella="sha256:x")
    suite = _suite(pin=pin, clasificacion=clasificacion, stop=stop)
    bloqueados: list[tuple[threading.Event, threading.Event, threading.Thread]] = []

    with leases_de_emision(**suite) as vistas:
        assert vistas.pin.valor == PinActivo(PIN, PROC)
        assert vistas.clasificacion.valor.contrato_de("CAP_X").identidad == "cap-x"
        assert vistas.stop.valor == EstadoStop(activo=False, huella="sha256:x")
        for proveedor in (pin, clasificacion, stop):
            intento = _intento_en_hilo(proveedor.lease_exclusivo)
            bloqueados.append(intento)
            _hasta_entrar_o_esperar(proveedor, intento[0])
            assert not intento[0].is_set(), "el escritor entro antes de persistir la decision"

    try:
        for entro, _soltar, _hilo in bloqueados:
            assert entro.wait(timeout=5.0), "el escritor no progreso al cerrar la frontera"
    finally:
        for _entro, soltar, _hilo in bloqueados:
            soltar.set()
    for _entro, _soltar, hilo in bloqueados:
        hilo.join(timeout=5.0)
        assert not hilo.is_alive()


def test_el_guard_nombra_el_checkpoint_de_auditoria_sin_confundirlo_con_block4() -> None:
    parametros = inspect.signature(leases_de_emision).parameters
    assert "rule_audit_checkpoint" in parametros
    assert "checkpoint" not in parametros


def test_el_guard_rechaza_el_checkpoint_real_de_block4(tmp_path) -> None:
    checkpoint_block4 = TrustedCheckpointStore(
        tmp_path / "block4-checkpoints.jsonl",
        bootstrap_receipt_path=tmp_path / "block4-receipt.json",
    )
    from policy.rule_authority.providers import (
        AlmacenCheckpointAuditoria,
        AlmacenCheckpointAuditoriaBloqueable,
    )

    assert not isinstance(checkpoint_block4, AlmacenCheckpointAuditoria)
    assert not isinstance(checkpoint_block4, AlmacenCheckpointAuditoriaBloqueable)
    _niega("checkpoint Rule Authority: no implementa", rule_audit_checkpoint=checkpoint_block4)


def test_a14_estres_ninguna_lectura_durante_exclusivo(intervalo_corto) -> None:
    """Estres ADICIONAL (la prueba determinista es test_a2_*): 0 lecturas con exclusivo vivo."""
    proveedor = PinFijo(PIN, PROC)
    violaciones = [0]
    exclusivo_vivo = [False]
    parar = threading.Event()

    def lector() -> None:
        while not parar.is_set():
            with proveedor.lease_compartido():
                if exclusivo_vivo[0]:
                    violaciones[0] += 1

    def escritor() -> None:
        while not parar.is_set():
            with proveedor.lease_exclusivo():
                exclusivo_vivo[0] = True
                time.sleep(0)
                exclusivo_vivo[0] = False

    hilos = [threading.Thread(target=lector, daemon=True) for _ in range(3)] + \
            [threading.Thread(target=escritor, daemon=True) for _ in range(2)]
    for h in hilos:
        h.start()
    time.sleep(0.35)
    parar.set()
    for h in hilos:
        h.join(timeout=5.0)
        assert not h.is_alive(), "un hilo de la carrera no termino (¿lock que no libera?)"
    assert violaciones[0] == 0, f"{violaciones[0]} lecturas durante exclusivo"


# ---------------------------------------------------------------------- pin

@pytest.mark.parametrize("mala", [
    "main", "HEAD", "origin/main", "heads/main", "REFS/heads/main", " refs/heads/main",
    "refs/heads/main ", "refs/heads/main\n", "refs/heads/", "refs/heads", "refs/heads/a b",
    "refs/remotes/origin/main", "refs/Heads/main", "refs/heads/../main", "refs/heads//x",
    "refs/heads/máin", "refs/tags/", "refs/heads/-x",
])
def test_x4_procedencia_fuera_de_la_forma_cerrada_niega(mala: str) -> None:
    pin = TrustedPolicyPin("jax", "0" * 40, "1" * 40, mala)   # el pin declara LA MISMA: solo la forma puede negar
    _niega("forma cerrada", pin=PinFijo(pin, mala))


@pytest.mark.parametrize("mala", ["refs/heads/a..b", "refs/heads/x.lock", "refs/heads/x.",
                                  "refs/heads/x.lock/y", "refs/tags/v1.", "refs/heads/a/b.lock"])
def test_procedencia_con_dos_puntos_lock_o_punto_final_niega(mala: str) -> None:
    pin = TrustedPolicyPin("jax", "0" * 40, "1" * 40, mala)
    _niega("forma cerrada", pin=PinFijo(pin, mala))


@pytest.mark.parametrize("buena", ["refs/heads/main", "refs/tags/v1.2.3", "refs/heads/feat/faro-f1.1_x"])
def test_x4_procedencia_cerrada_pasa(buena: str) -> None:
    pin = TrustedPolicyPin("jax", "0" * 40, "1" * 40, buena)
    exigir_contrato_de_emision(**_suite(pin=PinFijo(pin, buena)))


def test_x1_procedencia_sellada_distinta_de_la_del_pin_niega() -> None:
    _niega("difiere", pin=PinFijo(PIN, "refs/heads/otra"))     # ambas de forma cerrada, pero distintas
    _niega("forma cerrada", pin=PinFijo(PIN, ""))


def test_x1_pin_de_una_subclase_niega() -> None:
    class _PinHijo(TrustedPolicyPin):
        pass
    hijo = _PinHijo("jax", "0" * 40, "1" * 40, PROC)
    assert isinstance(hijo, TrustedPolicyPin)
    _niega("exactamente TrustedPolicyPin", pin=PinFijo(hijo, PROC))


@pytest.mark.parametrize("valor", [None, (PIN, PROC), "x", object(), PinActivo(None, PROC)])  # type: ignore[arg-type]
def test_vista_lease_con_valor_malo_niega_en_el_pin(valor: object) -> None:
    _niega("pin_activo", pin=_Crudo(valor))


# --------------------------------------------------------------------- stop

@pytest.mark.parametrize("estado", [
    EstadoStop(activo=None, huella="x"),       # type: ignore[arg-type]
    EstadoStop(activo=1, huella="x"),          # type: ignore[arg-type]
    EstadoStop(activo="false", huella="x"),    # type: ignore[arg-type]
    EstadoStop(activo=False, huella=""),
    EstadoStop(activo=False, huella=None),     # type: ignore[arg-type]
    EstadoStop(activo=False, huella=5),        # type: ignore[arg-type]
])
def test_x2_estado_stop_mal_formado_niega(estado: EstadoStop) -> None:
    _niega("EstadoStop mal formado", stop=_Crudo(estado))


@pytest.mark.parametrize("valor", [None, (False, "x"), "x"])
def test_x2_el_valor_del_lease_de_stop_debe_ser_estadostop(valor: object) -> None:
    _niega("no es EstadoStop", stop=_Crudo(valor))


def test_stop_ilegible_el_guard_niega_no_delega() -> None:
    """BLOCK r2: STOP ilegible NO es un estado valido que el guard deja pasar."""
    _niega("StopDesconocido", stop=StopIlegible())
    with pytest.raises(StopDesconocido):                      # el doble SI es ilegible de verdad
        with StopIlegible().lease_compartido():
            pass


def test_stop_activo_es_un_estado_valido_para_el_guard() -> None:
    """El guard verifica el CONTRATO del proveedor; que el STOP este activo lo
    traduce el kernel en DENY, no el guard."""
    exigir_contrato_de_emision(**_suite(stop=StopFijo(activo=True, huella="sha256:y")))


# --------------------------------------------------------------------- reloj

class _TzSinOffset(tzinfo):
    def utcoffset(self, _dt):
        return None

    def dst(self, _dt):
        return None

    def tzname(self, _dt):
        return None


@pytest.mark.parametrize("momento", [
    datetime(2026, 10, 6, 12, 0, 0),                                              # ingenua
    datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone(timedelta(hours=-6))),         # hora local consciente
    datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone(timedelta(minutes=1))),
    datetime(2026, 10, 6, 12, 0, 0, tzinfo=_TzSinOffset()),                        # utcoffset() None
])
def test_x3_reloj_que_no_es_utc_consciente_niega(momento: datetime) -> None:
    _niega("UTC consciente", reloj=_RelojCrudo(momento))


@pytest.mark.parametrize("momento", [None, "2026-10-06T12:00:00+00:00", 1759752000])
def test_reloj_que_no_entrega_datetime_niega(momento: object) -> None:
    _niega("datetime", reloj=_RelojCrudo(momento))


def test_reloj_utc_con_otra_clase_de_tz_pasa_y_el_que_lanza_niega() -> None:
    exigir_contrato_de_emision(**_suite(reloj=_RelojCrudo(datetime(2026, 10, 6, tzinfo=timezone(timedelta(0))))))

    class _Roto:
        def ahora(self):
            raise RelojRetrocedio("retrocedio")

        def marcar_emision(self, momento):
            return None
    _niega("ahora\\(\\) no sirve", reloj=_Roto())


def test_m4_reloj_ingenuo_del_doble_niega() -> None:
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

class _CatalogoPermisivo:
    """Responde SIEMPRE ``respuesta`` (un contrato REVERSIBLE, None...) a cualquier
    capability, desconocida incluida: el catalogo que §13 prohibe."""

    def __init__(self, respuesta: object) -> None:
        self._respuesta = respuesta

    def contrato_de(self, capability: str) -> object:
        return self._respuesta


class _CatalogoQueExplota:
    def contrato_de(self, capability: str) -> object:
        raise KeyError(capability)


@pytest.mark.parametrize("catalogo", [
    _CatalogoPermisivo(_contrato(ClaseCapability.REVERSIBLE)),
    _CatalogoPermisivo(_contrato(ClaseCapability.OBLIGATING)),
    _CatalogoPermisivo(None),
    _CatalogoQueExplota(),
])
def test_x5_capability_desconocida_que_no_niega_se_niega(catalogo: object) -> None:
    """Los catalogos abiertos no cruzan la frontera cerrada del lease."""
    _niega("CatalogoClasificacion cerrado", clasificacion=_Crudo(catalogo))


@pytest.mark.parametrize("valor", [None, "x", {"CAP_X": 1}])
def test_el_valor_del_lease_de_clasificacion_debe_ser_un_catalogo(valor: object) -> None:
    _niega("CatalogoClasificacion", clasificacion=_Crudo(valor))


def test_catalogo_cerrado_pasa_y_capability_ausente_niega() -> None:
    exigir_contrato_de_emision(**_suite(clasificacion=_Crudo(CatalogoFijo({}))))
    with pytest.raises(ClasificacionDesconocida):
        CatalogoFijo({"CAP_X": _contrato(ClaseCapability.OBLIGATING)}).contrato_de("__no_existe__")


class _CatalogoQueConoceLaCentinelaVieja:
    """Catalogo abierto: intenta distinguir un nombre reservado y dar fallback."""

    def contrato_de(self, capability: str) -> object:
        if capability == "__centinela_sin_clasificar__":
            raise ClasificacionDesconocida(capability)
        return _contrato(ClaseCapability.REVERSIBLE)


def test_catalogo_abierto_con_caso_especial_no_pasa() -> None:
    _niega("CatalogoClasificacion cerrado",
           clasificacion=_Crudo(_CatalogoQueConoceLaCentinelaVieja()))


class _CatalogoQueEvadePrefijoCentinela:
    """Elige excepcion por prefijo y da clase reversible al resto del mundo."""

    def contrato_de(self, capability: str) -> object:
        if capability.startswith("__centinela_sin_clasificar__"):
            raise ClasificacionDesconocida(capability)
        return _contrato(ClaseCapability.REVERSIBLE)


def test_un_catalogo_con_fallback_permisivo_no_pasa_por_evitar_el_sondeo() -> None:
    _niega("CatalogoClasificacion cerrado", clasificacion=_Crudo(_CatalogoQueEvadePrefijoCentinela()))


def test_a8_la_clase_es_enum_cerrado() -> None:
    with pytest.raises(ProveedorInvalido):
        ContratoCapability(identidad="c", version="v", clase="reversible?",   # type: ignore[arg-type]
                           unidad=None, moneda=None, forma_limites=FormaLimites.NINGUNA)
    with pytest.raises(ProveedorInvalido):
        ClasificacionFija({"D": None})                         # type: ignore[dict-item]


def test_m11_la_ausente_obliga_nunca_rebaja() -> None:
    with ClasificacionFija({}).lease_compartido() as vista:
        with pytest.raises(ClasificacionDesconocida):
            vista.valor.contrato_de("CAP_X")                   # type: ignore[attr-defined]


def test_sin_rebaja_entre_versiones_del_contrato() -> None:
    proveedor = ClasificacionFija({"CAP_X": _contrato(ClaseCapability.OBLIGATING)})
    with pytest.raises(ProveedorInvalido):
        proveedor.republicar("CAP_X", _contrato(ClaseCapability.REVERSIBLE))
    proveedor.republicar("CAP_X", _contrato(ClaseCapability.OBLIGATING))   # igual: ok
    with proveedor.lease_compartido() as vista:
        assert vista.valor.contrato_de("CAP_X").clase is ClaseCapability.OBLIGATING   # type: ignore[attr-defined]


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


def test_confirmar_es_el_log_contiene_el_head_exacto() -> None:
    """§11: no exige que sea el head ACTUAL (otra operacion pudo agregar un
    descendiente) y un head que nunca se publico NO se confirma."""
    checkpoints = CheckpointsEnMemoria()
    checkpoints.publicar("h1", anterior="")
    checkpoints.publicar("h2", anterior="h1")
    assert checkpoints.confirmar("h2") is True
    assert checkpoints.confirmar("h1") is True                 # contenido, aunque ya no sea el actual
    assert checkpoints.confirmar("h9") is False                # «siempre True» muere aqui
    assert CheckpointsEnMemoria().confirmar("h1") is False


def test_a13_carrera_hilos_cero_bifurcaciones(intervalo_corto) -> None:
    bifurcaciones = 0
    for _ in range(300):
        checkpoints = CheckpointsEnMemoria()
        checkpoints.publicar("h0", anterior="")
        exitosos: list[str] = []
        rechazos: list[CheckpointInvalido] = []
        barrera = threading.Barrier(2)

        def escritor(head: str) -> None:
            barrera.wait()
            try:
                checkpoints.publicar(head, anterior="h0")
                exitosos.append(head)
            except CheckpointInvalido as exc:
                rechazos.append(exc)

        hilos = [threading.Thread(target=escritor, args=(h,)) for h in ("hA", "hB")]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()
        assert len(exitosos) + len(rechazos) == 2
        if len(exitosos) == 2:
            bifurcaciones += 1
    assert bifurcaciones == 0, f"{bifurcaciones} bifurcaciones aceptadas"


class _CheckpointRecorder:
    """Checkpoint cuyo ``confirmar`` registra lo que le llaman y responde lo que se le diga."""

    def __init__(self, head: object = "h1", responde: object = True, explota: bool = False) -> None:
        self._head, self._responde, self._explota = head, responde, explota
        self.confirmados: list[object] = []

    def head_actual(self) -> object:
        return self._head

    def publicar(self, head: str, *, anterior: str) -> None:
        return None

    def confirmar(self, head: str) -> object:
        self.confirmados.append(head)
        if self._explota:
            raise OSError("log ilegible")
        return self._responde


def test_el_guard_llama_confirmar_con_el_head_exacto() -> None:
    rec = _CheckpointRecorder(head="h7")
    exigir_contrato_de_emision(**_suite(rule_audit_checkpoint=rec))
    assert rec.confirmados == ["h7"]


@pytest.mark.parametrize("responde", [False, None, "si", 1])
def test_el_guard_niega_si_el_log_no_confirma_el_head(responde: object) -> None:
    """Cae si el guard no llama confirmar, o lo acepta sin ser exactamente True."""
    _niega("no contiene el head", rule_audit_checkpoint=_CheckpointRecorder(responde=responde))


def test_el_guard_niega_si_confirmar_explota_o_el_head_es_ilegible() -> None:
    _niega("confirmar no sirve", rule_audit_checkpoint=_CheckpointRecorder(explota=True))
    _niega("nunca publicado", rule_audit_checkpoint=_CheckpointRecorder(head=""))
    _niega("nunca publicado", rule_audit_checkpoint=_CheckpointRecorder(head=None))


def test_a10_checkpoint_nunca_publicado_niega() -> None:
    with pytest.raises(ProveedorInvalido) as excinfo:
        exigir_contrato_de_emision(**_suite(checkpoint_publicado=False))
    assert "nunca publicado" in str(excinfo.value)


# -------------------------------------------------------------------- guard

def test_el_guard_pasa_la_suite_completa() -> None:
    exigir_contrato_de_emision(**_suite())                    # no lanza


def test_a4_el_guard_niega_mocks() -> None:
    with pytest.raises(ProveedorInvalido):
        exigir_contrato_de_emision(pin=Mock(), rule_audit_checkpoint=Mock(), stop=Mock(),
                                    reloj=Mock(), clasificacion=Mock())


class _LeaseRoto:
    """Pasa el isinstance (tiene los metodos) pero sus leases no sirven."""

    def lease_compartido(self):
        return None

    def lease_exclusivo(self):
        return None


class _LeaseVersionTrucha(_Crudo):
    def __init__(self, valor: object) -> None:
        super().__init__(valor, version=99)


@pytest.mark.parametrize("campo,valor", [
    ("pin", PinActivo(PIN, PROC)),
    ("stop", EstadoStop(activo=False, huella="x")),
    ("clasificacion", CatalogoFijo({})),
])
def test_a5_m5_m12_leases_rotos_o_con_vista_trucha_niegan(campo: str, valor: object) -> None:
    """Cada proveedor con el VALOR bueno: lo unico malo es el lease o la version."""
    _niega("lease", **{campo: _LeaseRoto()})
    _niega("VistaLease", **{campo: _LeaseVersionTrucha(valor)})


@pytest.mark.parametrize("campo", ["pin", "stop", "clasificacion"])
def test_el_proveedor_sin_leases_ni_siquiera_implementa_el_protocolo(campo: str) -> None:
    _niega("no implementa", **{campo: ProveedorSinLeases()})


def test_falta_cualquier_proveedor_niega() -> None:
    for campo in ("pin", "rule_audit_checkpoint", "stop", "reloj", "clasificacion"):
        _niega("falta el proveedor", **{campo: None})


def test_el_orden_global_de_adquisicion_es_el_del_diseno() -> None:
    assert ORDEN_ADQUISICION == ("pin_activo", "clasificacion", "stop", "permit_db",
                                 "authority_ledger_head", "audit_head")


class _PinSinExclusivo:
    """lease_compartido FUNCIONA; lease_exclusivo existe pero no es llamable."""

    def __init__(self) -> None:
        self._real = PinFijo(PIN, PROC)

    def lease_compartido(self):
        return self._real.lease_compartido()

    lease_exclusivo = None


def test_m12_el_guard_exige_lease_exclusivo_llamable() -> None:
    _niega("no implementa", pin=_PinSinExclusivo())


class _CheckpointAtributosMuertos:
    def head_actual(self) -> str:
        return "h1"

    publicar = None
    confirmar = None


def test_m6_el_guard_exige_publicar_y_confirmar_llamables() -> None:
    _niega("no implementa", rule_audit_checkpoint=_CheckpointAtributosMuertos())


class _RelojSinMarcar:
    def ahora(self) -> object:
        return T0

    marcar_emision = None


def test_m7_el_guard_exige_marcar_emision_llamable() -> None:
    _niega("no implementa", reloj=_RelojSinMarcar())


def test_a12_no_hay_lease_fabricable_en_produccion() -> None:
    import policy.rule_authority.providers as P
    assert not hasattr(P, "_Lease")
    falsa = VistaLease(valor=None, version=VersionMonotonica(999))
    assert falsa.valor is None                               # un dato, no una llave


def test_produccion_no_importa_pruebas() -> None:
    """policy/ y jax/ nunca importan tests/ ni los dobles (el modulo de produccion
    no depende de nada que viva en pruebas)."""
    for raiz in ("policy", "jax"):
        base = RAIZ / raiz
        if not base.is_dir():
            continue
        for archivo in base.rglob("*.py"):
            if "tests" in archivo.relative_to(base).parts or archivo.name.startswith("test_") \
                    or archivo.name.endswith("_test.py"):
                continue
            arbol = ast.parse(archivo.read_text(encoding="utf-8"))
            for nodo in ast.walk(arbol):
                nombres = []
                if isinstance(nodo, ast.Import):
                    nombres = [a.name for a in nodo.names]
                elif isinstance(nodo, ast.ImportFrom):
                    nombres = [nodo.module or ""]
                for nombre in nombres:
                    assert nombre.split(".")[0] != "tests" and "proveedores_dobles" not in nombre, \
                        f"{archivo}: importa {nombre}"
