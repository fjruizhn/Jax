"""Hechos del sistema del Ejecutor (Fase 2), derivados e inyectados en cada turno.

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §3.1 y §8 riesgo 4.

Tres reglas que estos tests fijan:
1. Cada hecho lleva su comando y su momento, a la vista.
2. Lo que no se pudo derivar se OMITE y se dice cuál y por qué.
3. Los hechos caducan (TTL). Y una captura truncada, o con código distinto
   de 0 (o sin código), no es un hecho: el error de la tarea 9 de U3 fue dar
   por bueno lo que llegó cortado.
"""
import pathlib

import pytest

from jax.ejecutor.captura import CapturaCompleta
from jax.ejecutor.hechos import (
    TTL_S_POR_DEFECTO, Fuente, Hecho, NoDerivado, bloque, derivar, vigente,
)

MOMENTO = "2026-09-16T10:00:00+00:00"


def _captura(comando, salida="up 38 minutes", codigo=0, truncada=False, motivos=()):
    return CapturaCompleta(
        maquina="local", comando=comando, codigo=codigo, salida=salida, stderr="",
        truncada=truncada, bytes_totales=len(salida.encode()), momento=MOMENTO,
        motivos_truncado=tuple(motivos))


def _correr_fijo(**campos):
    def falso(comando, maquina, tope_bytes, timeout_s):
        return _captura(comando, **campos)
    return falso


# --- los tres del plan ---------------------------------------------------

def test_un_hecho_vencido_no_entra_al_bloque():
    """TTL: un hecho de hace una hora presentado como actual es una mentira
    nueva (riesgo 4 del spec)."""
    h = Hecho(nombre="uptime", valor="38 min", comando="uptime -p", momento=MOMENTO)
    assert vigente(h, ahora="2026-09-16T10:00:30+00:00", ttl_s=60) is True
    assert vigente(h, ahora="2026-09-16T11:00:00+00:00", ttl_s=60) is False
    texto = bloque([h], ahora="2026-09-16T11:00:00+00:00", ttl_s=60)
    assert "38 min" not in texto
    assert "uptime" in texto and "vencido" in texto


def test_el_bloque_muestra_el_comando_y_la_hora_de_cada_hecho():
    h = Hecho(nombre="uptime", valor="38 min", comando="uptime -p", momento=MOMENTO)
    texto = bloque([h], ahora="2026-09-16T10:00:10+00:00")
    assert "uptime -p" in texto and MOMENTO in texto and "38 min" in texto


def test_un_hecho_que_no_se_pudo_derivar_se_omite_y_se_dice():
    """Desviación del plan: `no_derivados` lleva el MOTIVO, no sólo el nombre.
    Un nombre suelto es «desconocido sin decir por qué», que §3.1 prohíbe."""
    omitido = NoDerivado(nombre="disco", comando="df -h /", motivo="código de salida 1")
    texto = bloque([], ahora=MOMENTO, no_derivados=[omitido])
    assert "disco" in texto and "no se pudo derivar" in texto
    assert "código de salida 1" in texto and "df -h /" in texto


# --- el TTL compara instantes, no cadenas ----------------------------------

def test_el_ttl_compara_con_zona_no_como_cadena():
    """10:00:30Z es 04:00:30 en Tegucigalpa (UTC−6). Como cadenas, «04:00» es
    anterior a «10:00» y el hecho parecería del futuro o vencido."""
    h = Hecho(nombre="uptime", valor="38 min", comando="uptime -p",
              momento="2026-09-16T04:00:00-06:00")
    assert vigente(h, ahora="2026-09-16T10:00:30+00:00", ttl_s=60) is True
    assert vigente(h, ahora="2026-09-16T04:02:00-06:00", ttl_s=60) is False


def test_un_momento_sin_zona_no_es_vigente():
    """Sin zona no se sabe qué instante es: no se adivina."""
    h = Hecho(nombre="uptime", valor="38 min", comando="uptime -p",
              momento="2026-09-16T10:00:00")
    assert vigente(h, ahora="2026-09-16T10:00:10+00:00", ttl_s=60) is False


def test_un_momento_del_futuro_no_es_vigente():
    h = Hecho(nombre="uptime", valor="38 min", comando="uptime -p",
              momento="2026-09-16T10:05:00+00:00")
    assert vigente(h, ahora="2026-09-16T10:00:00+00:00", ttl_s=60) is False


def test_ahora_sin_zona_es_un_error_del_llamador():
    h = Hecho(nombre="uptime", valor="38 min", comando="uptime -p", momento=MOMENTO)
    with pytest.raises(ValueError):
        vigente(h, ahora="2026-09-16T10:00:10", ttl_s=60)


def test_el_ttl_por_defecto_arranca_en_60_segundos():
    assert TTL_S_POR_DEFECTO == 60


# --- derivar: lo que no es un hecho, no entra -------------------------------

def test_derivar_con_comandos_reales_lleva_procedencia():
    r = derivar([Fuente(nombre="eco", comando="echo hola")], maquina="local")
    assert r.no_derivados == ()
    [h] = r.hechos
    assert (h.nombre, h.valor, h.comando) == ("eco", "hola", "echo hola")
    assert h.momento.endswith("+00:00")


def test_una_captura_truncada_no_produce_hecho_y_se_dice_cual():
    """Tarea 9 de U3: dio por bueno lo que llegó cortado."""
    r = derivar([Fuente(nombre="numeros", comando="seq 1 100000")],
                maquina="local", tope_bytes=1024)
    assert r.hechos == ()
    [omitido] = r.no_derivados
    assert omitido.nombre == "numeros" and "truncada" in omitido.motivo
    assert "tope_bytes" in omitido.motivo


def test_un_codigo_distinto_de_cero_no_produce_hecho_y_se_dice_cual():
    r = derivar([Fuente(nombre="roto", comando="echo parcial; exit 3")], maquina="local")
    assert r.hechos == ()
    [omitido] = r.no_derivados
    assert omitido.nombre == "roto" and "3" in omitido.motivo


def test_sin_codigo_no_produce_hecho_y_se_dice_cual():
    """`codigo` None: el proceso no murió dentro del plazo."""
    r = derivar([Fuente(nombre="colgado", comando="sleep 999")], maquina="local",
                correr=_correr_fijo(codigo=None))
    assert r.hechos == ()
    [omitido] = r.no_derivados
    assert omitido.nombre == "colgado" and "sin código" in omitido.motivo


def test_la_marca_de_truncado_manda_aunque_el_codigo_sea_cero():
    r = derivar([Fuente(nombre="uptime", comando="uptime -p")], maquina="local",
                correr=_correr_fijo(truncada=True, motivos=("timeout",)))
    assert r.hechos == ()
    assert "timeout" in r.no_derivados[0].motivo


def test_una_salida_vacia_no_es_un_hecho():
    r = derivar([Fuente(nombre="nada", comando="true")], maquina="local")
    assert r.hechos == ()
    assert r.no_derivados[0].nombre == "nada"


def test_si_correr_lanza_se_omite_y_los_demas_siguen():
    def explota(comando, maquina, tope_bytes, timeout_s):
        if comando == "roto":
            raise OSError("sin shell")
        return _captura(comando)
    r = derivar([Fuente("a", "roto"), Fuente("b", "uptime -p")], maquina="local",
                correr=explota)
    assert [h.nombre for h in r.hechos] == ["b"]
    assert r.no_derivados[0].nombre == "a" and "sin shell" in r.no_derivados[0].motivo


def test_el_momento_del_hecho_es_el_de_la_captura():
    r = derivar([Fuente("uptime", "uptime -p")], maquina="local", correr=_correr_fijo())
    assert r.hechos[0].momento == MOMENTO


def test_el_inventario_base_cubre_lo_que_pide_el_spec():
    from jax.ejecutor.hechos import INVENTARIO_BASE
    nombres = {f.nombre for f in INVENTARIO_BASE}
    assert {"hostname", "uptime", "arranque", "so", "kernel", "disco"} <= nombres


def test_fuente_servicio_pregunta_desde_cuando_esta_activo():
    from jax.ejecutor.hechos import fuente_servicio
    f = fuente_servicio("jax-platform")
    assert "jax-platform" in f.comando and "ActiveEnterTimestamp" in f.comando


@pytest.mark.skipif(not pathlib.Path("/run/systemd/system").is_dir(),
                    reason="sin systemd corriendo (contenedor)")
def test_fuente_servicio_distingue_una_unidad_que_no_existe():
    """Medido en hall9000 el 2026-09-16: una unidad inexistente da código 0 y
    `ActiveState=inactive`, que se lee como «existe y está parada». Con
    LoadState la salida literal dice `not-found`."""
    from jax.ejecutor.hechos import fuente_servicio
    r = derivar([fuente_servicio("no-existe-hechos-jax-test")], maquina="local")
    assert "LoadState=not-found" in r.hechos[0].valor


def test_fuente_servicio_no_acepta_un_nombre_que_rompa_el_comando():
    from jax.ejecutor.hechos import fuente_servicio
    with pytest.raises(ValueError):
        fuente_servicio("x; rm -rf ~")


def test_un_valor_de_varias_lineas_no_se_confunde_con_otro_hecho():
    h = Hecho(nombre="disco", valor="Filesystem Size\n/dev/sda 100G",
              comando="df -h /", momento=MOMENTO)
    texto = bloque([h], ahora="2026-09-16T10:00:10+00:00")
    lineas = texto.splitlines()
    assert any(l.startswith("- disco = ") for l in lineas)
    assert not any(l.startswith("- /dev/sda") for l in lineas)
    assert "/dev/sda 100G" in texto
