"""Verificador de citas del Ejecutor (Fase 2).

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §3.3.
Puro: sin red, sin E/S, sin reloj. Lo corre tests-puros en CI.
"""
import pytest

from jax.ejecutor.cita import (
    DATO_FUERA_DE_LINEA, DATO_FUERA_DE_TEXTO, FUENTE_INEXISTENTE, FUENTE_TRUNCADA,
    NUMERO_SIN_RESPALDO, RESPALDADA, SIN_RESPALDO,
    Afirmacion, Captura, normalizar, numeros, verificar,
)

MAQUINA = "hall9000"
SALIDA_FREE = "               total        used        free\nMem:            89Gi        12Gi        70Gi"
CAPTURAS = [Captura(maquina=MAQUINA, comando="free -h", salida=SALIDA_FREE, stderr="", truncada=False)]


def test_una_linea_literal_esta_respaldada():
    a = Afirmacion(maquina=MAQUINA, texto="hay 89Gi de RAM", comando="free -h",
                   linea="Mem:            89Gi        12Gi        70Gi", dato="89Gi")
    assert verificar(a, CAPTURAS).estado == RESPALDADA


def test_una_linea_que_no_esta_en_la_salida_no_tiene_respaldo():
    a = Afirmacion(maquina=MAQUINA, texto="hay 128Gi de RAM", comando="free -h",
                   linea="Mem:           128Gi        12Gi        70Gi", dato="128Gi")
    assert verificar(a, CAPTURAS).estado == SIN_RESPALDO


def test_citar_un_comando_que_no_se_corrio_es_fuente_inexistente():
    a = Afirmacion(maquina=MAQUINA, texto="hay un disco sda", comando="lsblk", linea="sda", dato="sda")
    assert verificar(a, CAPTURAS).estado == FUENTE_INEXISTENTE


def test_una_captura_truncada_no_respalda_NADA_aunque_la_linea_este():
    """§2.4 y tarea 9 de U3: afirmó que todos los paquetes eran de `noble`
    habiendo visto 2 KB de una salida de 85,9 KB que nunca abrió. Si la
    salida vino cortada, no se mira el contenido: se rechaza antes."""
    capturas = [Captura(maquina=MAQUINA, comando="apt list", salida="paquete/noble 1.0", stderr="", truncada=True)]
    a = Afirmacion(maquina=MAQUINA, texto="es de noble", comando="apt list", linea="paquete/noble 1.0", dato="noble")
    assert verificar(a, capturas).estado == FUENTE_TRUNCADA


def test_los_espacios_no_deciden():
    a = Afirmacion(maquina=MAQUINA, texto="hay 89Gi", comando="free -h", linea="Mem: 89Gi 12Gi 70Gi", dato="89Gi")
    assert verificar(a, CAPTURAS).estado == RESPALDADA


def test_un_numero_parecido_NO_cuenta_como_respaldo():
    """Invención real de U3 (tarea 3): dijo «contexto 131.074» cuando su
    propia salida decía 131072. Si esto pasara, el verificador no sirve."""
    capturas = [Captura(maquina=MAQUINA, comando="ollama show", salida="context length 131072", stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, texto="el contexto es 131074", comando="ollama show",
                   linea="context length 131074", dato="131074")
    assert verificar(a, capturas).estado == SIN_RESPALDO


def test_las_mayusculas_SI_deciden():
    """No se normaliza mayúsculas: `Docker` y `docker` son datos distintos.

    La salida y la cita difieren SÓLO en la mayúscula. Corregido 2026-09-16
    al ejercitar la mutación `.lower()` del plan: la versión anterior
    comparaba `"node"` contra `"Docker"`, palabras distintas, y seguía verde
    con el verificador ignorando mayúsculas -- no probaba lo que dice."""
    capturas = [Captura(maquina=MAQUINA, comando="ss -ltnp", salida="LISTEN 0 4096 *:8188 users:((\"docker\"))", stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, texto="8188 es Docker", comando="ss -ltnp",
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
    a = Afirmacion(maquina=MAQUINA, texto="todos los paquetes son de noble", comando="apt list", linea=vacia, dato="noble")
    assert verificar(a, capturas).estado == SIN_RESPALDO


def test_si_el_comando_se_corrio_dos_veces_cuenta_cualquiera_de_las_capturas():
    """Hueco del plan: `verificar` contestaba con la PRIMERA captura del
    comando. Si la línea estaba en la segunda corrida, rechazaba trabajo
    bueno -- un falso positivo, lo que V2 prohíbe."""
    capturas = [
        Captura(maquina=MAQUINA, comando="systemctl is-active ollama", salida="activating", stderr="", truncada=False),
        Captura(maquina=MAQUINA, comando="systemctl is-active ollama", salida="active", stderr="", truncada=False),
    ]
    a = Afirmacion(maquina=MAQUINA, texto="ollama: active", comando="systemctl is-active ollama", linea="active", dato="active")
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
    a = Afirmacion(maquina=MAQUINA, texto="es de noble", comando="apt list", linea="paquete/noble 1.0", dato="noble")
    assert verificar(a, capturas).estado == FUENTE_TRUNCADA


def test_una_captura_completa_respalda_aunque_otra_corrida_haya_venido_truncada():
    capturas = [
        Captura(maquina=MAQUINA, comando="apt list", salida="paquete/noble 1.0", stderr="", truncada=True),
        Captura(maquina=MAQUINA, comando="apt list", salida="Listing...\npaquete/noble 1.0", stderr="", truncada=False),
    ]
    a = Afirmacion(maquina=MAQUINA, texto="es de noble", comando="apt list", linea="paquete/noble 1.0", dato="noble")
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
    a = Afirmacion(maquina=MAQUINA, texto="no tengo permiso: Permission denied", comando="cat /etc/shadow",
                   linea="cat: /etc/shadow: Permission denied", dato="Permission denied")
    assert verificar(a, capturas).estado == RESPALDADA


def test_una_linea_armada_pegando_el_final_de_stdout_con_el_principio_de_stderr_NO_se_respalda():
    """Si stdout no termina en salto de línea, concatenar los dos flujos
    fabrica una línea (`uso: 97%sudo: a password is required`) que la
    máquina nunca imprimió en ningún lado."""
    capturas = [Captura(maquina=MAQUINA, comando="df -h / ; sudo -n true",
                        salida="uso: 97%", stderr="sudo: a password is required\n",
                        truncada=False)]
    a = Afirmacion(maquina=MAQUINA, texto="inventada: uso: 97%sudo: a password is required", comando="df -h / ; sudo -n true",
                   linea="uso: 97%sudo: a password is required", dato="a password is required")
    assert verificar(a, capturas).estado == SIN_RESPALDO


def test_una_captura_truncada_no_respalda_ni_con_la_linea_en_stderr():
    """El truncado se mira ANTES que el contenido, también para stderr."""
    capturas = [Captura(maquina=MAQUINA, comando="apt update", salida="",
                        stderr="E: Could not get lock", truncada=True)]
    a = Afirmacion(maquina=MAQUINA, texto="el lock está tomado: Could not get lock", comando="apt update",
                   linea="E: Could not get lock", dato="Could not get lock")
    assert verificar(a, capturas).estado == FUENTE_TRUNCADA


def test_el_mismo_comando_corrido_en_OTRA_maquina_es_fuente_inexistente():
    """Sin esto, un `free -h` de otra máquina respalda una afirmación sobre
    ésta. No se corrió ESE comando EN ESA máquina."""
    capturas = [Captura(maquina="atemai", comando="free -h", salida=SALIDA_FREE,
                        stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, texto="hay 89Gi de RAM", comando="free -h",
                   linea="Mem:            89Gi        12Gi        70Gi", dato="89Gi")
    assert verificar(a, capturas).estado == FUENTE_INEXISTENTE


def test_una_captura_truncada_de_OTRA_maquina_no_convierte_el_veredicto_en_truncada():
    """La captura de otra máquina no cuenta para nada: ni para respaldar ni
    para decir que la fuente vino cortada."""
    capturas = [Captura(maquina="atemai", comando="free -h", salida=SALIDA_FREE,
                        stderr="", truncada=True)]
    a = Afirmacion(maquina=MAQUINA, texto="hay 89Gi de RAM", comando="free -h",
                   linea="Mem:            89Gi        12Gi        70Gi", dato="89Gi")
    assert verificar(a, capturas).estado == FUENTE_INEXISTENTE


@pytest.mark.parametrize("vacia", ["", "   "])
def test_una_afirmacion_que_no_dice_de_que_maquina_viene_no_se_respalda(vacia):
    """Una máquina vacía no identifica nada: si una captura también viniera
    sin máquina, `"" == ""` respaldaría una afirmación sin procedencia."""
    capturas = [Captura(maquina=vacia, comando="free -h", salida=SALIDA_FREE,
                        stderr="", truncada=False)]
    a = Afirmacion(maquina=vacia, texto="hay 89Gi de RAM", comando="free -h",
                   linea="Mem:            89Gi        12Gi        70Gi", dato="89Gi")
    assert verificar(a, capturas).estado == SIN_RESPALDO


# --- Ligadura afirmación ↔ cita (decisión de Fernando, 2026-09-16, con GO) ---
# El verificador anterior nunca leía `texto`: comprobaba que la línea citada
# existiera, no que respaldara lo afirmado («el servidor está en Marte» citando
# `free -h` salía respaldada; V1 medido = 0 de 11). Ahora la afirmación lleva
# `dato` y se exige: dato no vacío, dato literal en la línea, dato literal en el
# texto, y CADA número del texto presente en la línea citada.

LINEA_MEM = "Mem:            89Gi        12Gi        70Gi"
LINEA_CABECERA = "               total        used        free"


@pytest.mark.parametrize("linea", [LINEA_MEM, LINEA_CABECERA])
@pytest.mark.parametrize("dato", ["Marte", "89Gi", "total", "el servidor"])
def test_el_servidor_esta_en_Marte_no_se_respalda_con_ninguna_cita_real(linea, dato):
    a = Afirmacion(maquina=MAQUINA, texto="el servidor está en Marte", comando="free -h",
                   linea=linea, dato=dato)
    assert verificar(a, CAPTURAS).estado != RESPALDADA


def test_un_dato_que_no_esta_en_la_linea_citada_es_dato_fuera_de_linea():
    a = Afirmacion(maquina=MAQUINA, texto="el servidor está en Marte", comando="free -h",
                   linea=LINEA_MEM, dato="Marte")
    v = verificar(a, CAPTURAS)
    assert v.estado == DATO_FUERA_DE_LINEA
    assert "Marte" in v.motivo


def test_un_dato_verificado_no_blanquea_otro_numero_del_texto():
    """El defecto de la opción B que Fernando descartó: el dato está en la
    línea, pero la persona lee el TEXTO, y el texto dice 512 TB."""
    a = Afirmacion(maquina=MAQUINA, texto="hay 89Gi de RAM, o sea 512 TB", comando="free -h",
                   linea=LINEA_MEM, dato="89Gi")
    v = verificar(a, CAPTURAS)
    assert v.estado == NUMERO_SIN_RESPALDO
    assert "512" in v.motivo


def test_la_invencion_real_de_U3_131074_contra_la_linea_real_131072():
    capturas = [Captura(maquina=MAQUINA, comando="ollama show", salida="context length 131072",
                        stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, texto="el contexto es 131074", comando="ollama show",
                   linea="context length 131072", dato="131074")
    assert verificar(a, capturas).estado == DATO_FUERA_DE_LINEA


def test_131074_tampoco_entra_escondido_en_el_texto_junto_al_dato_real():
    capturas = [Captura(maquina=MAQUINA, comando="ollama show", salida="context length 131072",
                        stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, texto="contexto 131072 (unos 131,074 tokens)",
                   comando="ollama show", linea="context length 131072", dato="131072")
    assert verificar(a, capturas).estado == NUMERO_SIN_RESPALDO


def test_prosa_en_espanol_con_el_dato_literal_de_stderr_SI_se_respalda():
    """Lo que resuelve los falsos positivos de V2: la prosa puede estar en
    otro idioma mientras el dato sea literal y no haya números sueltos."""
    capturas = [Captura(maquina=MAQUINA, comando="sudo -n true", salida="",
                        stderr="sudo: a password is required\n", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, texto="sudo pide contraseña: a password is required",
                   comando="sudo -n true", linea="sudo: a password is required",
                   dato="a password is required")
    assert verificar(a, capturas).estado == RESPALDADA


@pytest.mark.parametrize("vacio", ["", "   ", "\t\n"])
def test_un_dato_vacio_no_se_respalda(vacio):
    """Tercer bypass por vacío (después de la cita y la máquina): `""` está
    dentro de cualquier línea y de cualquier texto."""
    a = Afirmacion(maquina=MAQUINA, texto="hay 89Gi de RAM", comando="free -h",
                   linea=LINEA_MEM, dato=vacio)
    assert verificar(a, CAPTURAS).estado == SIN_RESPALDO


def test_un_dato_que_esta_en_la_linea_pero_no_en_el_texto_es_dato_fuera_de_texto():
    a = Afirmacion(maquina=MAQUINA, texto="la máquina tiene poca memoria libre", comando="free -h",
                   linea=LINEA_MEM, dato="70Gi")
    assert verificar(a, CAPTURAS).estado == DATO_FUERA_DE_TEXTO


def test_el_dato_se_compara_con_la_misma_normalizacion_de_espacios_y_nada_mas():
    a = Afirmacion(maquina=MAQUINA, texto="free dice Mem: 89Gi  12Gi", comando="free -h",
                   linea=LINEA_MEM, dato="Mem:   89Gi 12Gi")
    assert verificar(a, CAPTURAS).estado == RESPALDADA
    b = Afirmacion(maquina=MAQUINA, texto="hay 89gi", comando="free -h", linea=LINEA_MEM, dato="89gi")
    assert verificar(b, CAPTURAS).estado == DATO_FUERA_DE_LINEA


def test_la_ligadura_se_exige_aunque_la_fuente_haya_venido_truncada():
    """Una afirmación incoherente consigo misma se rechaza por eso, antes de
    mirar las capturas: su veredicto no depende de qué se corrió."""
    capturas = [Captura(maquina=MAQUINA, comando="free -h", salida=SALIDA_FREE, stderr="", truncada=True)]
    a = Afirmacion(maquina=MAQUINA, texto="hay 89Gi de RAM, o sea 512 TB", comando="free -h",
                   linea=LINEA_MEM, dato="89Gi")
    assert verificar(a, capturas).estado == NUMERO_SIN_RESPALDO


# --- Qué es un «número» (regla 4) ---
# Número = secuencia maximal de dígitos, que puede llevar separadores internos
# `.` `,` `:` `-` `/` SIEMPRE entre dos dígitos. Se compara como token entero,
# con igualdad de cadena: ni subcadena ni valor numérico.

@pytest.mark.parametrize("texto,esperado", [
    ("3001", ("3001",)),
    ("131.074", ("131.074",)),
    ("131,074", ("131,074",)),
    ("Ubuntu 24.04.5 LTS", ("24.04.5",)),
    ("puerto 3001.", ("3001",)),              # el punto final no es separador
    ("1, 2 y 3", ("1", "2", "3")),            # coma seguida de espacio: separa
    ("hay 89Gi", ("89",)),                    # la unidad pegada no protege al número
    ("512TB", ("512",)),
    ("uso 5%", ("5",)),
    ("127.0.0.1:8084", ("127.0.0.1:8084",)),  # IP y puerto van juntos
    ("6.8.0-139-generic", ("6.8.0-139",)),
    ("2026-09-16", ("2026-09-16",)),
    ("12/24 núcleos", ("12/24",)),
    ("up 7:02", ("7:02",)),
    ("sin números", ()),
])
def test_numeros_tokeniza(texto, esperado):
    assert numeros(texto) == esperado


@pytest.mark.parametrize("texto,linea,dato", [
    # prefijo de otro número
    ("context length 131", "context length 131072", "context length"),
    # otro formato es otro literal
    ("context length 131.072", "context length 131072", "context length"),
    # IP de una columna, puerto de otra: `:` los ata
    ("LISTEN en 172.16.20.11:3001", "tcp LISTEN 0 511 172.16.20.11:8080 0.0.0.0:3001", "LISTEN"),
    # el 16 suelto de la hora no respalda la fecha: `-` la ata
    ("backup del 2026-09-16", "backup 2026-09-17 16:00", "backup"),
])
def test_un_numero_del_texto_tiene_que_estar_ENTERO_en_la_linea(texto, linea, dato):
    capturas = [Captura(maquina=MAQUINA, comando="c", salida=linea, stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, texto=texto, comando="c", linea=linea, dato=dato)
    assert verificar(a, capturas).estado == NUMERO_SIN_RESPALDO


@pytest.mark.parametrize("texto,linea,dato", [
    # Hallado al re-medir U3 (2026-09-16): `ss` imprime `0.0.0.0:8188`. Si `:`
    # sólo atara, «el puerto 8188» nunca tendría respaldo en la línea que lo
    # prueba, y 3 de las 4 etiquetas inventadas de la tarea 5 salían
    # «rechazadas» por este artefacto, no por la etiqueta.
    ("puerto 8188 en uso", "LISTEN 0 4096 0.0.0.0:8188 0.0.0.0:*", "8188"),
    ("escucha en 0.0.0.0", "LISTEN 0 4096 0.0.0.0:8188 0.0.0.0:*", "0.0.0.0"),
    ("kernel 6.8.0", "Linux 6.8.0-139-generic x86_64", "6.8.0"),
    ("empezó a las 16:00", "backup 2026-09-17 16:00:05", "16:00"),
])
def test_un_tramo_de_la_linea_separado_por_dos_puntos_guion_o_barra_respalda(texto, linea, dato):
    capturas = [Captura(maquina=MAQUINA, comando="c", salida=linea, stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, texto=texto, comando="c", linea=linea, dato=dato)
    assert verificar(a, capturas).estado == RESPALDADA


@pytest.mark.parametrize("texto,linea,dato", [
    # `.` y `,` NO parten: son una sola cantidad (miles, decimales, versión, IP).
    ("context length 131", "context length 131.072", "context length"),
    ("Ubuntu 24.04", "Ubuntu 24.04.5 LTS", "Ubuntu"),
    ("Linux 139", "Linux 6.8.0-139.139-generic", "Linux"),
])
def test_un_tramo_separado_por_punto_o_coma_NO_respalda(texto, linea, dato):
    capturas = [Captura(maquina=MAQUINA, comando="c", salida=linea, stderr="", truncada=False)]
    a = Afirmacion(maquina=MAQUINA, texto=texto, comando="c", linea=linea, dato=dato)
    assert verificar(a, capturas).estado == NUMERO_SIN_RESPALDO


# --- Límites conocidos, registrados y SIN arreglar (strict: si pasan, avisa) ---

@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "LÍMITE de la regla elegida (hallado 2026-09-16 al implementarla): las reglas "
    "2 y 3 las cumple CUALQUIER subcadena común a texto y línea, hasta una letra. "
    "Si el texto no tiene números, sólo la regla 4 no lo ata a nada. Contra un "
    "modelo que elige mal `dato` el verificador ata números, no prosa."))
def test_LIMITE_un_dato_trivial_no_deberia_respaldar_Marte():
    a = Afirmacion(maquina=MAQUINA, texto="el servidor está en Marte", comando="free -h",
                   linea=LINEA_MEM, dato="M")
    assert verificar(a, CAPTURAS).estado != RESPALDADA


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "LÍMITE: un número escrito con palabras no es un número para la regla 4."))
def test_LIMITE_un_numero_escrito_con_palabras_no_se_detecta():
    a = Afirmacion(maquina=MAQUINA, texto="hay 89Gi de RAM, o sea quinientos TB",
                   comando="free -h", linea=LINEA_MEM, dato="89Gi")
    assert verificar(a, CAPTURAS).estado != RESPALDADA
