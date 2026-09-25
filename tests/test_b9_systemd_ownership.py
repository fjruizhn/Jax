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
        # `Unit=` en [Timer] es OPCIONAL: sin él, systemd dispara el .service del
        # MISMO nombre por convención (systemd.timer(5)) -- exactamente lo que
        # jax-memory-worker.timer/jax-memory-synthesis.timer hacen instalados en
        # producción (ops/versionar-drop-ins, 2026-09-25: el repo pasó a ser
        # copia byte a byte de lo instalado, que no trae la línea). Lo que este
        # control protege de verdad es que, SI alguien escribe un `Unit=`
        # explícito, apunte al servicio correcto -- no que la línea exista.
        assert "Unit=" not in timer or f"Unit={name}.service" in timer, (
            f"{name}.timer fija Unit= a otro servicio -- revisar cuál dispara"
        )


def test_extraction_worker_does_not_launch_embedding_work():
    source = (Path(__file__).resolve().parents[1] / "jax" / "memory" / "worker.py").read_text(encoding="utf-8")
    run_once = source[source.index("async def run_once"):]
    assert "await _recalcular_embeddings_en_ceros(db)" not in run_once
