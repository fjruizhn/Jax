# tests/test_ejecutor_cuenta_de_servicio.py
"""Los servicios de JAX no corren como el operador (decisión de Fernando, 2026-09-17): corren
como la cuenta de servicio `jaxsvc`, y por eso `/etc/jax/.env` puede dejar de ser legible por
cualquier proceso de `fruiz`. Este control ata las unidades y los instaladores a esa cuenta:
un `User=fruiz` o un `install -o fruiz` que vuelva deja al servicio sin poder escribir lo suyo
(o, peor, devuelve el secreto al operador) y nadie se entera hasta el despliegue.

La cuenta de la JAULA (`JAX_EJECUTOR_CUENTA`, axioma) es otra cosa y no se toca acá.
"""
import re
from pathlib import Path

import pytest

from manifiesto_arranque import configuracion_efectiva

OPS = Path(__file__).resolve().parents[1] / "ops" / "ejecutor"
CUENTA_DE_SERVICIO = "jaxsvc"
# `ejecutor-vigia@.service` salió de acá (ronda 5, auditoría adversarial 2026-09-22): la
# unidad se RETIRÓ -- código muerto, nunca arrancó un vigía real (ver DEUDA.md). El vigía
# corre como subproceso directo de `jax-platform` (`fruiz`), no como `jaxsvc`; este control
# ya no le aplica.
#
# Las 4 unidades del manifiesto de arranque (auditoría escalón 3, m1: antes
# sólo cubría jax-ejecutor-proxy.service -- extendida a las 4 para que
# las-manos y los dos workers de memoria también queden bajo este control).
UNIDADES = (
    "jax-las-manos.service",
    "jax-memory-worker.service",
    "jax-memory-synthesis.service",
    "jax-ejecutor-proxy.service",
)
INSTALADORES = ("instalar_contratos.sh", "instalar_vigia.sh", "instalar_registro_y_cerco.sh")

# `_configuracion_efectiva` vive en tests/manifiesto_arranque.py (compartida
# con tests/test_arranque_instalado.py) desde la auditoría escalón 3 (m1):
# ahí se resuelve la unidad base + drop-ins de CUALQUIER unidad del
# manifiesto, con Environment= parseado con shlex.split (multi-asignación,
# vacío borra la lista) y separado por sección [Service] -- lo que antes
# vivía acá duplicado y sólo sabía buscar en ops/ejecutor/.


@pytest.mark.parametrize("nombre", UNIDADES)
def test_las_unidades_corren_como_la_cuenta_de_servicio(nombre):
    """Desde ops/versionar-drop-ins (2026-09-25) la unidad base en el repo es
    el fragmento CRUDO tal como está instalado en /etc (`User=fruiz`, sin
    `Environment=HOME=`) -- el `User=jaxsvc`/`HOME` propio llegan por
    `cuenta-de-servicio.conf`. Mirar sólo la base ya no prueba nada real:
    pasaría igual si alguien borrara el drop-in. Por eso este control mira
    la configuración EFECTIVA -- exactamente lo que corre -- y no sólo el
    archivo base."""
    simples, _ = configuracion_efectiva(nombre)
    assert simples.get("User") == CUENTA_DE_SERVICIO, (
        f"{nombre}: la configuración EFECTIVA (unidad base + drop-ins, último valor gana) "
        f"no corre como {CUENTA_DE_SERVICIO} -- User efectivo: {simples.get('User')!r}"
    )


@pytest.mark.parametrize("nombre", UNIDADES)
def test_las_unidades_le_dan_un_hogar_propio_a_la_cuenta(nombre):
    """Sin HOME propio, ssh busca known_hosts en el del operador y el turno muere con 255."""
    _, entorno = configuracion_efectiva(nombre)
    assert entorno.get("HOME") == "/var/lib/jaxsvc", (
        f"{nombre}: la configuración EFECTIVA no fija HOME propio -- HOME efectivo: "
        f"{entorno.get('HOME')!r}"
    )


@pytest.mark.parametrize("nombre", UNIDADES)
def test_los_archivos_de_unidad_no_apuntan_al_checkout_de_trabajo(nombre):
    """Auditoría escalón 3, M1: el PYTHONPATH efectivo de jax-ejecutor-proxy,
    jax-memory-worker y jax-memory-synthesis apuntaba a /home/fruiz/jax -- el
    checkout de TRABAJO de un agente, en rama ajena, no el de producción.
    Ningún valor de la unidad base + sus drop-ins (ni WorkingDirectory=, ni
    ExecStart=, ni PYTHONPATH dentro de Environment=, nada) puede mencionar
    /home/fruiz -- lo que corre en producción tiene que salir siempre de
    /srv/jax-prod/jax.

    ACOTACIÓN (auditoría escalón 3, ronda 2, MAJOR-2): esto cubre SÓLO los
    archivos de unidad (lo que arma este control). El PROCESO real además
    hereda `EnvironmentFile=/etc/jax/.env`, compartido por las 4 unidades, y
    ESE archivo sí tiene hoy claves con valores de /home/fruiz
    (JAX_AUDIT_LOG_PATH, JAX_REPO_BASE, JAX_MISSIONS_DIR, JAX_CONFIG_PATH,
    JAX_WORKSPACE_DIR -- nombres de clave confirmados con
    `sudo -n grep -oE` sobre el archivo real, nunca sus valores). Arreglar
    esas claves es una tarea aparte, de quien las declaró; este test NO
    afirma nada sobre la configuración EFECTIVA completa del proceso, sólo
    sobre lo que este árbol versiona y audita: los archivos de unidad."""
    simples, entorno = configuracion_efectiva(nombre)
    ofensores = {
        clave: valor
        for clave, valor in {**simples, **{f"Environment:{k}": v for k, v in entorno.items()}}.items()
        if "/home/fruiz" in valor
    }
    assert not ofensores, f"{nombre}: los archivos de unidad todavía apuntan a /home/fruiz: {ofensores!r}"


def _texto_con_delegados(nombre: str) -> str:
    """El texto del instalador Y el de cualquier script de `ops/ejecutor/` que invoque
    (B-2, ronda 4, 2026-09-22: `instalar_vigia.sh` delegó la creación del directorio de
    misiones -- antes un `install -d` inline -- a `preparar_directorio_misiones.sh`,
    que es lo que ahora de verdad decide el dueño). Sin esto, el control deja de ver el
    `-o jaxsvc` real apenas alguien extrae una línea a su propio script -- que es
    justamente el tipo de refactor que NO debería hacer que este control deje de
    proteger nada."""
    texto = (OPS / nombre).read_text()
    for delegado in re.findall(r'"\$REPO/ops/ejecutor/([\w.-]+\.sh)"', texto):
        texto += "\n" + (OPS / delegado).read_text()
    return texto


@pytest.mark.parametrize("nombre", INSTALADORES)
def test_los_instaladores_crean_los_directorios_para_la_cuenta_de_servicio(nombre):
    texto = _texto_con_delegados(nombre)
    assert "-o fruiz" not in texto, f"{nombre} sigue creando directorios del operador"
    assert "chown fruiz" not in texto, f"{nombre} le devuelve el dueño al operador"
    # `-o jaxsvc` (install) o `chown jaxsvc:jaxsvc` (ronda 7,
    # preparar_directorio_misiones.sh: setfacl atómico sin ventana, ver ese script) --
    # las dos formas dejan a jaxsvc como dueño, que es lo único que este control pide.
    assert f"-o {CUENTA_DE_SERVICIO}" in texto or f"chown {CUENTA_DE_SERVICIO}:" in texto


def test_la_unidad_del_vigia_no_existe_y_nadie_la_instala():
    """Ronda 5 (auditoría adversarial 2026-09-22): `ejecutor-vigia@.service` se retiró --
    código muerto, el camino real (`abrir_vigia`) siempre lanzó el vigía como subproceso
    directo. Guarda contra que vuelva sola (un merge viejo, una copia a mano) -- no contra
    que se la MENCIONE (los comentarios que explican el retiro sí la nombran, a propósito)."""
    assert not (OPS / "ejecutor-vigia@.service").exists()
    for nombre in INSTALADORES:
        lineas_de_codigo = [l for l in _texto_con_delegados(nombre).splitlines() if not l.strip().startswith("#")]
        assert not any("ejecutor-vigia@" in l for l in lineas_de_codigo), \
            f"{nombre} vuelve a INSTALAR la unidad retirada (fuera de un comentario)"
