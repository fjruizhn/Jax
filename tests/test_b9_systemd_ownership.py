from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "config" / "systemd"


def test_every_b9_scheduled_job_has_a_systemd_owner():
    expected = {
        "jax-memory-worker": "jax.memory.worker",
        "jax-memory-embedding": "jax.memory.embedding_worker",
        "jax-memory-synthesis": "jax.memory.synthesis_worker",
        "jax-memory-lifecycle": "jax.memory.lifecycle_worker",
        "jax-memory-vector-health": "jax.memory.embedding_worker --health",
    }
    for name, module in expected.items():
        service = (ROOT / f"{name}.service").read_text(encoding="utf-8")
        timer = (ROOT / f"{name}.timer").read_text(encoding="utf-8")
        assert f"-m {module}" in service
        assert f"Unit={name}.service" in timer


# --- Auditoria 2026-10-05, MAJOR-2 y menores: las unidades del repo = lo que corre en produccion ---

def _opciones(texto: str, seccion: str, clave: str) -> list[str]:
    """Valores de `clave` dentro de `[seccion]` (systemd admite claves repetidas)."""
    actual, valores = None, []
    for linea in texto.splitlines():
        linea = linea.strip()
        if linea.startswith("[") and linea.endswith("]"):
            actual = linea[1:-1]
        elif actual == seccion and linea.startswith(f"{clave}="):
            valores.append(linea.split("=", 1)[1])
    return valores


def _unidades_memoria() -> list[Path]:
    return sorted(ROOT.glob("jax-memory-*.service"))


def test_hay_cinco_unidades_de_memoria():
    # Si alguien agrega o quita una, que las pruebas de abajo se enteren y se revisen.
    assert [p.stem for p in _unidades_memoria()] == [
        "jax-memory-embedding", "jax-memory-lifecycle", "jax-memory-synthesis",
        "jax-memory-vector-health", "jax-memory-worker"]


def test_toda_unidad_jax_memory_avisa_si_falla():
    """Sin OnFailure= un timer que falla cada 20 minutos no se entera nadie (MAJOR-2)."""
    for unit in _unidades_memoria():
        assert _opciones(unit.read_text(encoding="utf-8"), "Unit", "OnFailure") == ["aviso-fallo@%n.service"], unit.name


def test_embedding_lifecycle_y_vector_health_validan_el_checkout_como_worker_y_synthesis():
    pre = "/usr/local/sbin/jax-checkout-de-produccion-sano.sh /srv/jax-prod/jax"
    for name in ("jax-memory-embedding", "jax-memory-lifecycle", "jax-memory-vector-health"):
        texto = (ROOT / f"{name}.service").read_text(encoding="utf-8")
        assert _opciones(texto, "Service", "ExecStartPre") == [pre], name


def test_el_timer_de_synthesis_corre_a_las_0430_una_vez_al_dia():
    """DeepSeek se paga por corrida: lo instalado en produccion es 04:30 diario, no cada hora."""
    timer = (ROOT / "jax-memory-synthesis.timer").read_text(encoding="utf-8")
    assert _opciones(timer, "Timer", "OnCalendar") == ["*-*-* 04:30:00"]


# --- Drop-ins de worker y synthesis (2026-10-05): el repo reproduce lo instalado en /etc/systemd/system ---
#
# Los dos llevan en produccion tres drop-ins fuera del repo: el checkout de produccion con su freno
# (`jax-checkout-de-produccion-sano.sh`), la cuenta de servicio y un PYTHONPATH que apunta a
# /srv/jax-prod/jax (el 2026-09-25 apuntaba al checkout de TRABAJO de un agente, jax#274). Sin versionarlos,
# reinstalar desde el repo los pierde en silencio. Copiados byte a byte de /etc/systemd/system.

import subprocess  # noqa: E402

import pytest  # noqa: E402

_CON_DROPINS = {
    "jax-memory-worker": "jax.memory.worker",
    "jax-memory-synthesis": "jax.memory.synthesis_worker",
}
_DROPINS = ("checkout-de-produccion.conf", "cuenta-de-servicio.conf", "z-pythonpath.conf")
_RAIZ_DE_PRODUCCION = "/srv/jax-prod/jax"


def _dropin(unidad: str, nombre: str) -> str:
    ruta = ROOT / f"{unidad}.service.d" / nombre
    assert ruta.is_file(), f"falta {ruta.relative_to(ROOT.parents[1])}: la instalacion desde el repo lo perderia"
    return ruta.read_text(encoding="utf-8")


@pytest.mark.parametrize("unidad", sorted(_CON_DROPINS))
def test_el_dropin_del_checkout_fija_directorio_freno_y_ejecutable_de_produccion(unidad):
    texto = _dropin(unidad, "checkout-de-produccion.conf")
    assert _opciones(texto, "Service", "WorkingDirectory") == [_RAIZ_DE_PRODUCCION]
    assert _opciones(texto, "Service", "ExecStartPre") == [
        f"/usr/local/sbin/jax-checkout-de-produccion-sano.sh {_RAIZ_DE_PRODUCCION}"]
    # `ExecStart=` vacio primero: resetea el de la unidad, si no systemd rechaza dos ExecStart en un oneshot.
    assert _opciones(texto, "Service", "ExecStart") == [
        "", f"{_RAIZ_DE_PRODUCCION}/.venv/bin/python -m {_CON_DROPINS[unidad]}"]


@pytest.mark.parametrize("unidad", sorted(_CON_DROPINS))
def test_el_dropin_de_la_cuenta_corre_como_jaxsvc_con_su_home(unidad):
    texto = _dropin(unidad, "cuenta-de-servicio.conf")
    assert _opciones(texto, "Service", "User") == ["jaxsvc"]
    assert _opciones(texto, "Service", "Group") == ["jaxsvc"]
    home = _opciones(texto, "Service", "Environment")
    assert len(home) == 1 and home[0].startswith("HOME=/") and "/home/" not in home[0]


@pytest.mark.parametrize("unidad", sorted(_CON_DROPINS))
def test_el_pythonpath_apunta_al_checkout_de_produccion_y_nunca_al_de_trabajo(unidad):
    texto = _dropin(unidad, "z-pythonpath.conf")
    assert _opciones(texto, "Service", "Environment") == [f"PYTHONPATH={_RAIZ_DE_PRODUCCION}"]
    # `z-` ordena el archivo ultimo: gana sobre cualquier PYTHONPATH de los demas drop-ins.
    assert sorted(p.name for p in (ROOT / f"{unidad}.service.d").glob("*.conf"))[-1] == "z-pythonpath.conf"


def test_install_memory_scope_instala_los_dropins_de_worker_y_synthesis(tmp_path):
    """Se ejecuta el instalador de verdad contra un destino de prueba (nunca /etc)."""
    destino = tmp_path / "etc-systemd-system"
    destino.mkdir()
    r = subprocess.run(
        ["bash", str(ROOT / "install-memory-scope.sh"), "--instalar-obsoleto"],
        env={"PATH": "/usr/bin:/bin", "JAX_SYSTEMD_SRC": str(ROOT), "JAX_SYSTEMD_DEST": str(destino)},
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    for unidad in _CON_DROPINS:
        for nombre in _DROPINS:
            instalado = destino / f"{unidad}.service.d" / nombre
            assert instalado.is_file(), f"el instalador no dejo {instalado.name} de {unidad}"
            assert instalado.read_bytes() == (ROOT / f"{unidad}.service.d" / nombre).read_bytes()
        assert (destino / f"{unidad}.service").is_file() and (destino / f"{unidad}.timer").is_file()
    # No habilita ni arranca nada: solo copia.
    assert not list(destino.rglob("*.wants"))


def test_install_memory_scope_sin_bandera_no_hace_nada(tmp_path):
    destino = tmp_path / "etc"
    destino.mkdir()
    r = subprocess.run(["bash", str(ROOT / "install-memory-scope.sh")],
                       env={"PATH": "/usr/bin:/bin", "JAX_SYSTEMD_SRC": str(ROOT), "JAX_SYSTEMD_DEST": str(destino)},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and not list(destino.iterdir())
