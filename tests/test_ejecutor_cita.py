"""Verificador de citas del Ejecutor (Fase 2).

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §2.0 y §3.3.
Puro: sin red, sin E/S, sin reloj. Lo corre tests-puros en CI.

DECISIÓN de Fernando (2026-09-16, §2.0): el Ejecutor no escribe prosa. La
afirmación es `(maquina, comando, linea, dato)` y lo que ve la persona lo arma
`presentar`, no el modelo.
"""
from dataclasses import fields

import pytest

from jax.ejecutor.cita import (
    DATO_FUERA_DE_LINEA, FUENTE_INEXISTENTE, FUENTE_TRUNCADA, RESPALDADA, SIN_RESPALDO,
    Afirmacion, Captura, normalizar, presentar, verificar,
)

MAQUINA = "hall9000"
SALIDA_FREE = "               total        used        free\nMem:            89Gi        12Gi        70Gi"
CAPTURAS = [Captura(maquina=MAQUINA, comando="free -h", salida=SALIDA_FREE, stderr="", truncada=False)]
LINEA_MEM = "Mem:            89Gi        12Gi        70Gi"


def _una(linea_salida, linea, dato, comando="c", truncada=False):
    capturas = [Captura(maquina=MAQUINA, comando=comando, salida=linea_salida, stderr="", truncada=truncada)]
    return verificar(Afirmacion(maquina=MAQUINA, comando=comando, linea=linea, dato=dato), capturas)


# --- El contrato: no hay campo de prosa ---

def test_la_afirmacion_no_tiene_campo_de_prosa():
    """§2.0: «No hay campo de texto libre escrito por el modelo»."""
    assert [f.name for f in fields(Afirmacion)] == ["maquina", "comando", "linea", "dato"]


def test_no_se_puede_construir_una_afirmacion_con_texto():
    with pytest.raises(TypeError):
        Afirmacion(maquina=MAQUINA, texto="el servidor está en Marte", comando="free -h",
                   linea=LINEA_MEM, dato="89Gi")


# --- La fuente ---

def test_una_linea_literal_esta_respaldada():
    a = Afirmacion(maquina=MAQUINA, comando="free -h", linea=LINEA_MEM, dato="89Gi")
    assert verificar(a, CAPTURAS).estado == RESPALDADA


def test_una_linea_que_no_esta_en_la_salida_no_tiene_respaldo():
    a = Afirmacion(maquina=MAQUINA, comando="free -h",
                   linea="Mem:           128Gi        12Gi        70Gi", dato="128Gi")
    assert verificar(a, CAPTURAS).estado == SIN_RESPALDO


def test_citar_un_comando_que_no_se_corrio_es_fuente_inexistente():
    a = Afirmacion(maquina=MAQUINA, comando="lsblk", linea="sda", dato="sda")
    assert verificar(a, CAPTURAS).estado == FUENTE_INEXISTENTE


def test_una_captura_truncada_no_respalda_NADA_aunque_la_linea_este():
    """§2.4 y tarea 9 de U3: afirmó que todos los paquetes eran de `noble`
    habiendo visto 2 KB de una salida de 85,9 KB que nunca abrió. Si la
    salida vino cortada, no se mira el contenido: se rechaza antes."""
    capturas = [Captura(maquina=MAQUINA, comando="apt list", salida="paquete/noble 1.0", stderr="", truncada=True)]
    a = Afirmacion(maquina=MAQUINA, comando="apt list", linea="paquete/noble 1.0", dato="noble")
    assert verificar(a, capturas).estado == FUENTE_TRUNCADA


def test_los_espacios_no_deciden():
    a = Afirmacion(maquina=MAQUINA, comando="free -h", linea="Mem: 89Gi 12Gi 70Gi", dato="89Gi")
    assert verificar(a, CAPTURAS).estado == RESPALDADA


def test_un_numero_parecido_NO_cuenta_como_respaldo():
    """Invención real de U3 (tarea 3): dijo «contexto 131.074» cuando su
    propia salida decía 131072. Si esto pasara, el verificador no sirve."""
    capturas = [Captura(maquina=MAQUINA, comando="ollama show", salida="context length 131072", stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, comando="ollama show", linea="context length 131074", dato="131074")
    assert verificar(a, capturas).estado == SIN_RESPALDO


def test_las_mayusculas_SI_deciden():
    """No se normaliza mayúsculas: `Docker` y `docker` son datos distintos.

    La salida y la cita difieren SÓLO en la mayúscula. Corregido 2026-09-16
    al ejercitar la mutación `.lower()` del plan: la versión anterior
    comparaba `"node"` contra `"Docker"`, palabras distintas, y seguía verde
    con el verificador ignorando mayúsculas -- no probaba lo que dice."""
    capturas = [Captura(maquina=MAQUINA, comando="ss -ltnp", salida="LISTEN 0 4096 *:8188 users:((\"docker\"))", stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, comando="ss -ltnp",
                   linea="LISTEN 0 4096 *:8188 users:((\"Docker\"))", dato="Docker")
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
    capturas = [Captura(maquina=MAQUINA, comando="apt list", salida="Listing...\n\npaquete/noble 1.0", stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, comando="apt list", linea=vacia, dato="noble")
    assert verificar(a, capturas).estado == SIN_RESPALDO


def test_si_el_comando_se_corrio_dos_veces_cuenta_cualquiera_de_las_capturas():
    """Hueco del plan: `verificar` contestaba con la PRIMERA captura del
    comando. Si la línea estaba en la segunda corrida, rechazaba trabajo
    bueno -- un falso positivo, lo que V2 prohíbe."""
    capturas = [
        Captura(maquina=MAQUINA, comando="systemctl is-active ollama", salida="activating", stderr="", truncada=False),
        Captura(maquina=MAQUINA, comando="systemctl is-active ollama", salida="active", stderr="", truncada=False),
    ]
    a = Afirmacion(maquina=MAQUINA, comando="systemctl is-active ollama", linea="active", dato="active")
    assert verificar(a, capturas).estado == RESPALDADA


def test_con_varias_capturas_una_truncada_sigue_sin_respaldar_NADA():
    """El arreglo de arriba no puede abrir la puerta de §2.4: si la línea sólo
    aparece en la captura truncada, NO se respalda, aunque haya otra completa
    del mismo comando. Recorrer todas las capturas mirando el contenido antes
    que el truncado dejaría pasar exactamente la tarea 9 de U3."""
    capturas = [
        Captura(maquina=MAQUINA, comando="apt list", salida="Listing...", stderr="", truncada=False),
        Captura(maquina=MAQUINA, comando="apt list", salida="paquete/noble 1.0", stderr="", truncada=True),
    ]
    a = Afirmacion(maquina=MAQUINA, comando="apt list", linea="paquete/noble 1.0", dato="noble")
    assert verificar(a, capturas).estado == FUENTE_TRUNCADA


def test_una_captura_completa_respalda_aunque_otra_corrida_haya_venido_truncada():
    capturas = [
        Captura(maquina=MAQUINA, comando="apt list", salida="paquete/noble 1.0", stderr="", truncada=True),
        Captura(maquina=MAQUINA, comando="apt list", salida="Listing...\npaquete/noble 1.0", stderr="", truncada=False),
    ]
    a = Afirmacion(maquina=MAQUINA, comando="apt list", linea="paquete/noble 1.0", dato="noble")
    assert verificar(a, capturas).estado == RESPALDADA


# --- Contrato 2026-09-16 (decisiones del orquestador, autorizadas por Fernando) ---
# 1 · stderr se puede citar, pero stdout y stderr no se mezclan.
# 2 · La afirmación dice de qué máquina viene; respalda sólo máquina + comando.


def test_una_linea_que_solo_esta_en_stderr_queda_respaldada():
    """Spec §3.3: «literal en el stdout/stderr». En U3 la tarea 10 -- la única
    que se comportó bien -- lo hizo mostrando `Permission denied` y el `sudo`
    pidiendo contraseña, que viven en stderr. Si stderr no fuera citable, la
    conducta correcta quedaría sin respaldo."""
    capturas = [Captura(maquina=MAQUINA, comando="cat /etc/shadow", salida="",
                        stderr="cat: /etc/shadow: Permission denied\n", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, comando="cat /etc/shadow",
                   linea="cat: /etc/shadow: Permission denied", dato="Permission denied")
    assert verificar(a, capturas).estado == RESPALDADA


def test_una_linea_armada_pegando_el_final_de_stdout_con_el_principio_de_stderr_NO_se_respalda():
    """Si stdout no termina en salto de línea, concatenar los dos flujos
    fabrica una línea (`uso: 97%sudo: a password is required`) que la
    máquina nunca imprimió en ningún lado."""
    capturas = [Captura(maquina=MAQUINA, comando="df -h / ; sudo -n true",
                        salida="uso: 97%", stderr="sudo: a password is required\n",
                        truncada=False)]
    a = Afirmacion(maquina=MAQUINA, comando="df -h / ; sudo -n true",
                   linea="uso: 97%sudo: a password is required", dato="a password is required")
    assert verificar(a, capturas).estado == SIN_RESPALDO


def test_una_captura_truncada_no_respalda_ni_con_la_linea_en_stderr():
    """El truncado se mira ANTES que el contenido, también para stderr."""
    capturas = [Captura(maquina=MAQUINA, comando="apt update", salida="",
                        stderr="E: Could not get lock", truncada=True)]
    a = Afirmacion(maquina=MAQUINA, comando="apt update",
                   linea="E: Could not get lock", dato="Could not get lock")
    assert verificar(a, capturas).estado == FUENTE_TRUNCADA


def test_el_mismo_comando_corrido_en_OTRA_maquina_es_fuente_inexistente():
    """Sin esto, un `free -h` de otra máquina respalda una afirmación sobre
    ésta. No se corrió ESE comando EN ESA máquina."""
    capturas = [Captura(maquina="atemai", comando="free -h", salida=SALIDA_FREE,
                        stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, comando="free -h", linea=LINEA_MEM, dato="89Gi")
    assert verificar(a, capturas).estado == FUENTE_INEXISTENTE


def test_una_captura_truncada_de_OTRA_maquina_no_convierte_el_veredicto_en_truncada():
    """La captura de otra máquina no cuenta para nada: ni para respaldar ni
    para decir que la fuente vino cortada."""
    capturas = [Captura(maquina="atemai", comando="free -h", salida=SALIDA_FREE,
                        stderr="", truncada=True)]
    a = Afirmacion(maquina=MAQUINA, comando="free -h", linea=LINEA_MEM, dato="89Gi")
    assert verificar(a, capturas).estado == FUENTE_INEXISTENTE


@pytest.mark.parametrize("vacia", ["", "   "])
def test_una_afirmacion_que_no_dice_de_que_maquina_viene_no_se_respalda(vacia):
    """Una máquina vacía no identifica nada: si una captura también viniera
    sin máquina, `"" == ""` respaldaría una afirmación sin procedencia."""
    capturas = [Captura(maquina=vacia, comando="free -h", salida=SALIDA_FREE,
                        stderr="", truncada=False)]
    a = Afirmacion(maquina=vacia, comando="free -h", linea=LINEA_MEM, dato="89Gi")
    assert verificar(a, capturas).estado == SIN_RESPALDO


# --- El dato, ligado a la línea ---


def test_un_dato_que_no_esta_en_la_linea_citada_es_dato_fuera_de_linea():
    a = Afirmacion(maquina=MAQUINA, comando="free -h", linea=LINEA_MEM, dato="Marte")
    v = verificar(a, CAPTURAS)
    assert v.estado == DATO_FUERA_DE_LINEA
    assert "Marte" in v.motivo


def test_la_invencion_real_de_U3_131074_contra_la_linea_real_131072():
    assert _una("context length 131072", "context length 131072", "131074").estado == DATO_FUERA_DE_LINEA


@pytest.mark.parametrize("vacio", ["", "   ", "\t\n"])
def test_un_dato_vacio_no_se_respalda(vacio):
    """Tercer bypass por vacío (después de la cita y la máquina): `""` está
    dentro de cualquier línea."""
    a = Afirmacion(maquina=MAQUINA, comando="free -h", linea=LINEA_MEM, dato=vacio)
    assert verificar(a, CAPTURAS).estado == SIN_RESPALDO


def test_el_dato_se_compara_con_la_misma_normalizacion_de_espacios_y_nada_mas():
    a = Afirmacion(maquina=MAQUINA, comando="free -h", linea=LINEA_MEM, dato="Mem:   89Gi 12Gi")
    assert verificar(a, CAPTURAS).estado == RESPALDADA
    b = Afirmacion(maquina=MAQUINA, comando="free -h", linea=LINEA_MEM, dato="89gi")
    assert verificar(b, CAPTURAS).estado == DATO_FUERA_DE_LINEA


def test_la_ligadura_se_exige_aunque_la_fuente_haya_venido_truncada():
    """Una afirmación incoherente consigo misma se rechaza por eso, antes de
    mirar las capturas: su veredicto no depende de qué se corrió."""
    capturas = [Captura(maquina=MAQUINA, comando="free -h", salida=SALIDA_FREE, stderr="", truncada=True)]
    a = Afirmacion(maquina=MAQUINA, comando="free -h", linea=LINEA_MEM, dato="512 TB")
    assert verificar(a, capturas).estado == DATO_FUERA_DE_LINEA


# --- Hallado 2026-09-16 al quitar la prosa: el dato no puede cortar un token ---
# Sin prosa, el `dato` es lo único que el modelo escribe. Con subcadena pura,
# `active` sale de `inactive` y `3107` de `131072`: un dato falso con una línea
# verdadera. Token = corrida de letras, o número con `.`/`,` internos (miles,
# decimales, versión, IP son UNA cantidad). `:` `-` `/` `_`, espacios y el paso
# letra↔dígito separan.


@pytest.mark.parametrize("linea,dato", [
    ("inactive (dead)", "active"),
    ("context length 131072", "3107"),
    ("context length 131072", "131"),
    ("context length 131.072", "131"),
    ("context length 131.072", "072"),
    ("Ubuntu 24.04.5 LTS", "24.04"),
    ("Ubuntu 24.04.5 LTS", "4.04.5"),
    ("Mem: 89Gi", "M"),
    ("Mem: 89Gi", "9Gi"),
    ("unavailable", "available"),
])
def test_un_dato_que_corta_un_token_de_la_linea_no_se_respalda(linea, dato):
    assert _una(linea, linea, dato).estado == DATO_FUERA_DE_LINEA


@pytest.mark.parametrize("linea,dato", [
    ("LISTEN 0 4096 0.0.0.0:8188 0.0.0.0:*", "8188"),       # ss imprime ip:puerto
    ("LISTEN 0 4096 0.0.0.0:8188 0.0.0.0:*", "0.0.0.0"),
    ("Linux 6.8.0-139-generic x86_64", "6.8.0"),
    ("backup 2026-09-17 16:00:05", "16:00"),
    ("acl/noble-updates,now 2.3.2 amd64", "noble"),
    ("Mem: 89Gi 12Gi", "89"),                                # letra↔dígito separa
    ("Mem: 89Gi 12Gi", "89Gi"),
    ("inactive active", "active"),                           # vale otra aparición
    ("/dev/nvme0n1p2 1863G 92G 1771G 5% /", "1863G"),
])
def test_un_dato_que_respeta_los_bordes_de_token_se_respalda(linea, dato):
    assert _una(linea, linea, dato).estado == RESPALDADA


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "LÍMITE (2026-09-16): el dato puede ser palabras enteras que, fuera de la línea, "
    "dicen lo contrario. `presentar` muestra la línea completa con el `No`; si la "
    "persona lee sólo el dato, el error es de lectura (§2.0). No lo cierra el "
    "verificador: lo mitiga la presentación."))
def test_LIMITE_un_dato_de_palabras_enteras_puede_soltar_la_negacion():
    linea = "No reboot required file found"
    assert _una(linea, linea, "reboot required").estado != RESPALDADA


# --- Lo que ve la persona lo arma el sistema ---


def test_el_veredicto_respaldado_trae_la_linea_tal_como_la_imprimio_la_maquina():
    """La cita se compara con espacios colapsados; lo que se muestra es la
    línea de la captura, no la versión que escribió el modelo."""
    a = Afirmacion(maquina=MAQUINA, comando="free -h", linea="Mem: 89Gi 12Gi 70Gi", dato="89Gi")
    v = verificar(a, CAPTURAS)
    assert v.estado == RESPALDADA
    assert v.linea_capturada == LINEA_MEM


def test_un_veredicto_no_respaldado_no_trae_linea():
    a = Afirmacion(maquina=MAQUINA, comando="free -h", linea=LINEA_MEM, dato="Marte")
    assert verificar(a, CAPTURAS).linea_capturada == ""


def test_presentar_muestra_dato_maquina_comando_y_la_linea_completa_literal():
    a = Afirmacion(maquina=MAQUINA, comando="free -h", linea=LINEA_MEM, dato="89Gi")
    p = presentar(a)
    assert p.splitlines() == [
        "dato: '89Gi'",
        "máquina: 'hall9000'",
        "comando: 'free -h'",
        f"línea: {LINEA_MEM!r}",
    ]


def test_presentar_no_recorta_una_linea_larga():
    """Nunca resume: la línea sale entera, por larga que sea."""
    larga = "x " * 5000 + "fin"
    a = Afirmacion(maquina=MAQUINA, comando="c", linea=larga, dato="fin")
    assert repr(larga) in presentar(a)


@pytest.mark.parametrize("campo", ["maquina", "comando", "linea", "dato"])
@pytest.mark.parametrize("colado", ["\n", "\r", " ", "\x1b[2K", "\x85"])
def test_presentar_no_deja_que_un_campo_fabrique_otra_linea_ni_borre_la_pantalla(campo, colado):
    """Un salto de línea o un escape de terminal dentro de un campo podría
    dibujar un `dato:` falso o borrar el `No` de la línea. Cada campo sale
    escapado: siempre cuatro líneas, sin caracteres de control."""
    valores = dict(maquina=MAQUINA, comando="c", linea="a b", dato="a")
    valores[campo] = valores[campo] + colado + "dato: 'falso'"
    p = presentar(Afirmacion(**valores))
    assert len(p.splitlines()) == 4
    assert "\x1b" not in p
