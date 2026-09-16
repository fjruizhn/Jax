"""Verificador de citas del Ejecutor (Fase 2).

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §3.3.
Puro: sin red, sin E/S, sin reloj. Lo corre tests-puros en CI.
"""
import pytest

from jax.ejecutor.cita import (
    FUENTE_INEXISTENTE, FUENTE_TRUNCADA, RESPALDADA, SIN_RESPALDO,
    Afirmacion, Captura, normalizar, verificar,
)

SALIDA_FREE = "               total        used        free\nMem:            89Gi        12Gi        70Gi"
CAPTURAS = [Captura(comando="free -h", salida=SALIDA_FREE, truncada=False)]


def test_una_linea_literal_esta_respaldada():
    a = Afirmacion(texto="hay 89Gi de RAM", comando="free -h",
                   linea="Mem:            89Gi        12Gi        70Gi")
    assert verificar(a, CAPTURAS).estado == RESPALDADA


def test_una_linea_que_no_esta_en_la_salida_no_tiene_respaldo():
    a = Afirmacion(texto="hay 128Gi de RAM", comando="free -h",
                   linea="Mem:           128Gi        12Gi        70Gi")
    assert verificar(a, CAPTURAS).estado == SIN_RESPALDO


def test_citar_un_comando_que_no_se_corrio_es_fuente_inexistente():
    a = Afirmacion(texto="algo", comando="lsblk", linea="sda")
    assert verificar(a, CAPTURAS).estado == FUENTE_INEXISTENTE


def test_una_captura_truncada_no_respalda_NADA_aunque_la_linea_este():
    """§2.4 y tarea 9 de U3: afirmó que todos los paquetes eran de `noble`
    habiendo visto 2 KB de una salida de 85,9 KB que nunca abrió. Si la
    salida vino cortada, no se mira el contenido: se rechaza antes."""
    capturas = [Captura(comando="apt list", salida="paquete/noble 1.0", truncada=True)]
    a = Afirmacion(texto="es de noble", comando="apt list", linea="paquete/noble 1.0")
    assert verificar(a, capturas).estado == FUENTE_TRUNCADA


def test_los_espacios_no_deciden():
    a = Afirmacion(texto="hay 89Gi", comando="free -h", linea="Mem: 89Gi 12Gi 70Gi")
    assert verificar(a, CAPTURAS).estado == RESPALDADA


def test_un_numero_parecido_NO_cuenta_como_respaldo():
    """Invención real de U3 (tarea 3): dijo «contexto 131.074» cuando su
    propia salida decía 131072. Si esto pasara, el verificador no sirve."""
    capturas = [Captura(comando="ollama show", salida="context length 131072", truncada=False)]
    a = Afirmacion(texto="el contexto es 131074", comando="ollama show",
                   linea="context length 131074")
    assert verificar(a, capturas).estado == SIN_RESPALDO


def test_las_mayusculas_SI_deciden():
    """No se normaliza mayúsculas: `Docker` y `docker` son datos distintos.

    La salida y la cita difieren SÓLO en la mayúscula. Corregido 2026-09-16
    al ejercitar la mutación `.lower()` del plan: la versión anterior
    comparaba `"node"` contra `"Docker"`, palabras distintas, y seguía verde
    con el verificador ignorando mayúsculas -- no probaba lo que dice."""
    capturas = [Captura(comando="ss -ltnp", salida="LISTEN 0 4096 *:8188 users:((\"docker\"))", truncada=False)]
    a = Afirmacion(texto="8188 es Docker", comando="ss -ltnp",
                   linea="LISTEN 0 4096 *:8188 users:((\"Docker\"))")
    assert verificar(a, capturas).estado == SIN_RESPALDO


def test_normalizar_colapsa_espacios_pero_no_toca_el_resto():
    assert normalizar("  Mem:   89Gi  ") == "Mem: 89Gi"
    assert normalizar("89 GiB") != normalizar("91 GB")


# --- Agregados 2026-09-16 al implementar la Task 1 (no estaban en el plan) ---


@pytest.mark.parametrize("vacia", ["", "   ", "\t"])
def test_una_cita_vacia_NO_se_respalda_con_una_linea_en_blanco(vacia):
    """Hueco del plan: `normalizar("")` y `normalizar("   ")` dan `""`, que
    es igual a cualquier línea en blanco de la salida. Una afirmación
    inventada con la línea vacía salía `respaldada` contra casi cualquier
    comando real (systemctl, apt, df con cabecera partida...)."""
    capturas = [Captura(comando="apt list", salida="Listing...\n\npaquete/noble 1.0", truncada=False)]
    a = Afirmacion(texto="todos los paquetes son de noble", comando="apt list", linea=vacia)
    assert verificar(a, capturas).estado == SIN_RESPALDO


def test_si_el_comando_se_corrio_dos_veces_cuenta_cualquiera_de_las_capturas():
    """Hueco del plan: `verificar` contestaba con la PRIMERA captura del
    comando. Si la línea estaba en la segunda corrida, rechazaba trabajo
    bueno -- un falso positivo, lo que V2 prohíbe."""
    capturas = [
        Captura(comando="systemctl is-active ollama", salida="activating", truncada=False),
        Captura(comando="systemctl is-active ollama", salida="active", truncada=False),
    ]
    a = Afirmacion(texto="ollama está activo", comando="systemctl is-active ollama", linea="active")
    assert verificar(a, capturas).estado == RESPALDADA


def test_con_varias_capturas_una_truncada_sigue_sin_respaldar_NADA():
    """El arreglo de arriba no puede abrir la puerta de §2.4: si la línea sólo
    aparece en la captura truncada, NO se respalda, aunque haya otra completa
    del mismo comando. Recorrer todas las capturas mirando el contenido antes
    que el truncado dejaría pasar exactamente la tarea 9 de U3."""
    capturas = [
        Captura(comando="apt list", salida="Listing...", truncada=False),
        Captura(comando="apt list", salida="paquete/noble 1.0", truncada=True),
    ]
    a = Afirmacion(texto="es de noble", comando="apt list", linea="paquete/noble 1.0")
    assert verificar(a, capturas).estado == FUENTE_TRUNCADA


def test_una_captura_completa_respalda_aunque_otra_corrida_haya_venido_truncada():
    capturas = [
        Captura(comando="apt list", salida="paquete/noble 1.0", truncada=True),
        Captura(comando="apt list", salida="Listing...\npaquete/noble 1.0", truncada=False),
    ]
    a = Afirmacion(texto="es de noble", comando="apt list", linea="paquete/noble 1.0")
    assert verificar(a, capturas).estado == RESPALDADA
