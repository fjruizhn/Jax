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
