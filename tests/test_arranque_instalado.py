"""Versionado del arranque de systemd que antes sólo vivía en /etc de
hall9000 (decisión de Fernando, 2026-09-25, opción A).

Dos partes:

(a) SIEMPRE corre, también en CI sin ningún host de producción cerca: la
    forma del repo es correcta -- cada archivo del manifiesto existe, los
    drop-ins apuntan a las rutas de producción correctas, y el guion de
    sanidad tiene el bit ejecutable EN GIT (no sólo en el filesystem de
    quien corrió el checkout).

(b) SÓLO en el host de producción (si existen /srv/jax-prod/jax y todas
    las rutas /etc del manifiesto): corre
    ops/verificar-arranque-instalado.sh de verdad y exige código 0. Fuera
    de ese host, skip con el motivo explícito.

El manifiesto es UNA sola lista, ops/manifiesto-arranque-instalado.tsv. La
lectura vive en tests/manifiesto_arranque.py (importada acá) y el propio
guion (ops/verificar-arranque-instalado.sh) la vuelve a leer del mismo
archivo por su cuenta, en shell -- dos listas que pudieran divergir serían
peor que una sola que puede fallar.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from manifiesto_arranque import MANIFIESTO, ROOT, leer_manifiesto as _leer_manifiesto

SCRIPT = ROOT / "ops" / "verificar-arranque-instalado.sh"

RUTA_SANIDAD = "/usr/local/sbin/jax-checkout-de-produccion-sano.sh"
RAIZ_PRODUCCION = "/srv/jax-prod/jax"


def test_el_manifiesto_existe_y_no_esta_vacio():
    assert MANIFIESTO.is_file()
    assert _leer_manifiesto(), "el manifiesto no puede estar vacío"


def test_cada_archivo_del_manifiesto_existe_en_el_repo():
    faltantes = [str(repo_abs) for repo_abs, _ in _leer_manifiesto() if not repo_abs.is_file()]
    assert not faltantes, f"faltan en el repo: {faltantes}"


def test_los_checkout_de_produccion_referencian_el_guion_de_sanidad():
    conf = [repo_abs for repo_abs, _ in _leer_manifiesto() if repo_abs.name == "checkout-de-produccion.conf"]
    assert conf, "se esperaba al menos un checkout-de-produccion.conf en el manifiesto"
    for c in conf:
        contenido = c.read_text(encoding="utf-8")
        assert RUTA_SANIDAD in contenido, f"{c} no referencia {RUTA_SANIDAD}"


def test_los_dropins_de_raiz_referencian_la_raiz_de_produccion():
    """checkout-de-produccion.conf y z-pythonpath.conf fijan rutas contra
    /srv/jax-prod/jax -- cuenta-de-servicio.conf NO (sólo fija User/Group/
    HOME, ver el test de abajo), así que no entra en este chequeo."""
    conf = [
        repo_abs for repo_abs, _ in _leer_manifiesto()
        if repo_abs.name in ("checkout-de-produccion.conf", "z-pythonpath.conf")
    ]
    assert conf, "se esperaban checkout-de-produccion.conf/z-pythonpath.conf en el manifiesto"
    for c in conf:
        contenido = c.read_text(encoding="utf-8")
        assert RAIZ_PRODUCCION in contenido, f"{c} no referencia {RAIZ_PRODUCCION}"


def test_cuenta_de_servicio_fija_jaxsvc_con_hogar_propio():
    conf = [repo_abs for repo_abs, _ in _leer_manifiesto() if repo_abs.name == "cuenta-de-servicio.conf"]
    assert conf, "se esperaba al menos un cuenta-de-servicio.conf en el manifiesto"
    for c in conf:
        contenido = c.read_text(encoding="utf-8")
        assert "User=jaxsvc" in contenido, f"{c} no fija User=jaxsvc"
        assert "Group=jaxsvc" in contenido, f"{c} no fija Group=jaxsvc"
        assert "Environment=HOME=/var/lib/jaxsvc" in contenido, f"{c} no fija HOME propio"


def test_z_pythonpath_fija_la_raiz_del_checkout_de_produccion():
    """Las 4 unidades con z-pythonpath.conf (las-manos, proxy del Ejecutor,
    memory-worker, memory-synthesis) -- auditoría escalón 3, M1: el
    PYTHONPATH efectivo de las últimas 3 apuntaba a /home/fruiz/jax, el
    checkout de TRABAJO de un agente, no el de producción."""
    zpps = [repo_abs for repo_abs, _ in _leer_manifiesto() if repo_abs.name == "z-pythonpath.conf"]
    assert len(zpps) == 4, f"se esperaban 4 z-pythonpath.conf en el manifiesto, hay {len(zpps)}: {zpps}"
    for zpp in zpps:
        contenido = zpp.read_text(encoding="utf-8")
        assert f"PYTHONPATH={RAIZ_PRODUCCION}" in contenido, f"{zpp} no fija PYTHONPATH con {RAIZ_PRODUCCION}"

    proxy_zpp = ROOT / "ops" / "ejecutor" / "jax-ejecutor-proxy.service.d" / "z-pythonpath.conf"
    assert proxy_zpp in zpps
    assert f"PYTHONPATH={RAIZ_PRODUCCION}:{RAIZ_PRODUCCION}/las_manos" in proxy_zpp.read_text(encoding="utf-8"), (
        "el proxy del Ejecutor necesita las_manos/ en el PYTHONPATH (facet_resolver, "
        "credential_resolver, db_connect_config son imports pelados que dependen de eso)"
    )


def test_el_guion_de_sanidad_tiene_bit_ejecutable_en_git():
    salida = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-s", "ops/sbin/jax-checkout-de-produccion-sano.sh"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert salida.startswith("100755"), f"modo en git no es ejecutable: {salida!r}"


def test_guion_de_verificacion_tiene_bit_ejecutable_en_git():
    salida = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-s", "ops/verificar-arranque-instalado.sh"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert salida.startswith("100755"), f"modo en git no es ejecutable: {salida!r}"


def _motivo_de_skip_fuera_de_produccion() -> str | None:
    """None si ESTA máquina es el host de producción de jax; si no, el
    motivo del skip.

    La máquina es producción SÍ Y SÓLO SI existe /srv/jax-prod/jax --
    auditoría escalón 3, M3: la versión anterior TAMBIÉN saltaba si faltaba
    algún archivo instalado del manifiesto, lo que le daba a cualquier
    drop-in sin instalar un skip silencioso en vez de un rojo, incluso en
    el host de producción real. Ser producción no depende de tener el
    arranque completo instalado -- es al revés: si sos producción y falta
    algo, ESO es lo que este test tiene que gritar."""
    if not Path(RAIZ_PRODUCCION).is_dir():
        return f"esta máquina no tiene {RAIZ_PRODUCCION} -- no es el host de producción de jax"
    return None


def test_verificar_arranque_instalado_da_cero_en_produccion():
    motivo = _motivo_de_skip_fuera_de_produccion()
    if motivo:
        pytest.skip(motivo)
    resultado = subprocess.run([str(SCRIPT)], capture_output=True, text=True)
    assert resultado.returncode == 0, (
        f"ops/verificar-arranque-instalado.sh salió {resultado.returncode}:\n"
        f"stdout: {resultado.stdout}\nstderr: {resultado.stderr}"
    )
