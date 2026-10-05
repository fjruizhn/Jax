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


def test_extraction_worker_does_not_launch_embedding_work():
    source = (Path(__file__).resolve().parents[1] / "jax" / "memory" / "worker.py").read_text(encoding="utf-8")
    run_once = source[source.index("async def run_once"):]
    assert "await _recalcular_embeddings_en_ceros(db)" not in run_once


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
