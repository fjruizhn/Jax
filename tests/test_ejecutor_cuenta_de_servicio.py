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

OPS = Path(__file__).resolve().parents[1] / "ops" / "ejecutor"
CUENTA_DE_SERVICIO = "jaxsvc"
# `ejecutor-vigia@.service` salió de acá (ronda 5, auditoría adversarial 2026-09-22): la
# unidad se RETIRÓ -- código muerto, nunca arrancó un vigía real (ver DEUDA.md). El vigía
# corre como subproceso directo de `jax-platform` (`fruiz`), no como `jaxsvc`; este control
# ya no le aplica.
UNIDADES = ("jax-ejecutor-proxy.service",)
INSTALADORES = ("instalar_contratos.sh", "instalar_vigia.sh", "instalar_registro_y_cerco.sh")


@pytest.mark.parametrize("nombre", UNIDADES)
def test_las_unidades_corren_como_la_cuenta_de_servicio(nombre):
    lineas = [l.strip() for l in (OPS / nombre).read_text().splitlines()]
    assert f"User={CUENTA_DE_SERVICIO}" in lineas, f"{nombre} no corre como {CUENTA_DE_SERVICIO}"
    assert not any(l.startswith("User=fruiz") for l in lineas)


@pytest.mark.parametrize("nombre", UNIDADES)
def test_las_unidades_le_dan_un_hogar_propio_a_la_cuenta(nombre):
    """Sin HOME propio, ssh busca known_hosts en el del operador y el turno muere con 255."""
    texto = (OPS / nombre).read_text()
    assert "Environment=HOME=/var/lib/jaxsvc" in texto


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
    assert f"-o {CUENTA_DE_SERVICIO}" in texto


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
