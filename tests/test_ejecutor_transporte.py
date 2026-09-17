"""Transporte del Ejecutor (Fase 2): lo que no cita, no sale.

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §2.1, §2.3,
§5 y V4 de §6. Plan: Task 5, adaptada a la DECISIÓN de §2.0 (sin prosa).
Puro: sin red ni E/S.
"""
import pytest

from jax.ejecutor import transporte
from jax.ejecutor.cita import (
    DATO_FUERA_DE_LINEA, FUENTE_TRUNCADA, RESPALDADA, SIN_RESPALDO, Afirmacion, Captura,
)

MAQUINA = "hall9000"
CAPTURAS = [Captura(maquina=MAQUINA, comando="uptime -p", salida="up   38 minutes\n", stderr="", truncada=False)]
BUENA = Afirmacion(maquina=MAQUINA, comando="uptime -p", linea="up 38 minutes", dato="38 minutes")
MALA = Afirmacion(maquina=MAQUINA, comando="uptime -p", linea="up 23 hours", dato="23 hours")


def test_solo_salen_las_afirmaciones_respaldadas():
    e = transporte.entregar([BUENA, MALA], CAPTURAS)
    assert [a.dato for a in e.respaldadas] == ["38 minutes"]
    assert [d.afirmacion for d in e.descartadas] == [MALA]


def test_cada_descartada_dice_por_que():
    fuera = Afirmacion(maquina=MAQUINA, comando="uptime -p", linea="up 38 minutes", dato="un día")
    e = transporte.entregar([MALA, fuera], CAPTURAS)
    assert [(d.estado, bool(d.motivo)) for d in e.descartadas] == [
        (SIN_RESPALDO, True), (DATO_FUERA_DE_LINEA, True)]


def test_las_salidas_crudas_se_entregan_SIEMPRE():
    """El piso de §2.3: si no puede citar nada, no queda mudo."""
    e = transporte.entregar([MALA], CAPTURAS)
    assert e.respaldadas == ()
    assert e.crudas == tuple(CAPTURAS)


def test_sin_afirmaciones_tambien_salen_las_crudas():
    e = transporte.entregar([], CAPTURAS)
    assert e.crudas == tuple(CAPTURAS)
    assert e.respaldadas == () and e.descartadas == ()


def test_las_crudas_salen_aunque_esten_truncadas_y_la_afirmacion_no():
    """Truncada no respalda afirmaciones (§2.4), pero se entrega igual: es
    lo que se recolectó, marcado."""
    truncada = [Captura(maquina=MAQUINA, comando="uptime -p", salida="up 38 minutes", stderr="", truncada=True)]
    e = transporte.entregar([BUENA], truncada)
    assert e.crudas == tuple(truncada)
    assert e.respaldadas == ()
    assert [d.estado for d in e.descartadas] == [FUENTE_TRUNCADA]


def test_la_respaldada_lleva_la_linea_tal_como_la_imprimio_la_maquina():
    """El modelo citó `up 38 minutes`; la máquina imprimió `up   38 minutes`.
    Lo que se entrega (y lo que `presentar` muestra) es lo de la máquina."""
    e = transporte.entregar([BUENA], CAPTURAS)
    assert e.respaldadas[0].linea == "up   38 minutes"


def test_presentar_la_entrega_usa_la_linea_de_la_maquina():
    e = transporte.entregar([BUENA], CAPTURAS)
    assert transporte.presentar(e) == ["dato: '38 minutes'\nmáquina: 'hall9000'\n"
                                       "comando: 'uptime -p'\nlínea: 'up   38 minutes'"]


def test_un_generador_de_afirmaciones_se_recorre_una_vez_y_bien():
    e = transporte.entregar((a for a in [BUENA, MALA]), iter(CAPTURAS))
    assert len(e.respaldadas) == 1 and len(e.descartadas) == 1
    assert e.crudas == tuple(CAPTURAS)


# --- V4: fail-closed ---


def test_v4_si_el_verificador_falla_NO_sale_ninguna_afirmacion(monkeypatch):
    """Fail-closed: un verificador caído no puede volverse un pase libre.

    Revienta en la SEGUNDA llamada: la primera afirmación ya salió
    `respaldada`, y aun así no se entrega. Un verificador que falló a mitad
    de turno no deja sano nada de ese turno."""
    real = transporte.cita.verificar
    llamadas = []

    def explota_en_la_segunda(afirmacion, capturas):
        llamadas.append(afirmacion)
        if len(llamadas) == 2:
            raise RuntimeError("verificador caído")
        return real(afirmacion, capturas)

    monkeypatch.setattr(transporte.cita, "verificar", explota_en_la_segunda)
    otra_buena = Afirmacion(maquina=MAQUINA, comando="uptime -p", linea="up 38 minutes", dato="38")
    e = transporte.entregar([BUENA, otra_buena], CAPTURAS)
    assert real(BUENA, CAPTURAS).estado == RESPALDADA  # la primera SÍ estaba respaldada
    assert e.respaldadas == ()
    assert e.crudas == tuple(CAPTURAS)
    assert [d.estado for d in e.descartadas] == [transporte.VERIFICADOR_CAIDO] * 2
    assert all("RuntimeError" in d.motivo for d in e.descartadas)


def test_v4_si_las_afirmaciones_no_se_pueden_leer_salen_las_crudas_y_ninguna_afirmacion():
    def rota():
        yield BUENA
        raise OSError("se cortó la lectura")
    e = transporte.entregar(rota(), CAPTURAS)
    assert e.respaldadas == ()
    assert e.crudas == tuple(CAPTURAS)


@pytest.mark.parametrize("veredicto_raro", ["RESPALDADA", "respaldada ", None, ""])
def test_solo_el_estado_exacto_respaldada_deja_salir(monkeypatch, veredicto_raro):
    """Lista de permitidos, no de prohibidos: un estado desconocido descarta."""
    monkeypatch.setattr(transporte.cita, "verificar",
                        lambda a, c: transporte.cita.Veredicto(veredicto_raro, "raro"))
    e = transporte.entregar([BUENA], CAPTURAS)
    assert e.respaldadas == ()
