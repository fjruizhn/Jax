#!/usr/bin/env python3
"""E2E real (R4, Tasks 1-5 aplicadas) -- no mockea httpx ni credencial.
Requiere jax-las-manos.service corriendo y Ollama con qwen3-coder:30b
cargado. Verifica el criterio de aceptación 1 y 2 del spec: Qwen ejecuta
una tarea de código real (no chat), Kimi vía Pipeline con motor explícito
vs auto.

Corre desde /home/fruiz/jax con:
  PYTHONPATH=/home/fruiz/jax .venv/bin/python jacobs/_pipeline_motor_e2e_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import httpx

LAS_MANOS = "http://127.0.0.1:7777"


async def _dispatch_and_wait(*, caller, capability, motor, prompt, timeout=60):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(f"{LAS_MANOS}/motor/dispatch", json={
            "caller": caller, "capability": capability, "motor": motor, "prompt": prompt,
        })
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(2)
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{LAS_MANOS}/motor/job/{job_id}")
            job = r.json()
        if job["status"] in ("completed", "failed", "rejected"):
            return job
    raise TimeoutError(f"job {job_id} no completó en {timeout}s")


async def test_qwen_ejecuta_tarea_de_codigo_real():
    job = await _dispatch_and_wait(
        caller="jacobs", capability="generate", motor="jax_local",
        prompt="Escribí una función Python que sume dos números, con type hints.",
    )
    assert job["status"] == "completed", job
    print(f"OK Qwen: {job.get('result_summary', '')[:100]}")


async def test_kimi_con_motor_explicito_via_pipeline():
    job = await _dispatch_and_wait(
        caller="jacobs", capability="pipeline_analysis", motor="kimi",
        prompt="Confirmá en una frase que estás operativo.",
    )
    assert job["status"] == "completed", job
    print(f"OK Kimi explícito: {job.get('result_summary', '')[:100]}")


async def test_capability_generate_sin_motor_explicito_resuelve_auto():
    """motor=None -- MotorPolicy._resolve_motor() elige el primero
    habilitado de allowed_motors (generate: kimi, ada, jax_local en ese
    orden de priority)."""
    job = await _dispatch_and_wait(
        caller="jacobs", capability="generate", motor=None,
        prompt="Respondé solo con la palabra: listo.",
    )
    assert job["status"] == "completed", job
    print(f"OK auto: motor resuelto por competencia, job {job}")


async def main():
    await test_qwen_ejecuta_tarea_de_codigo_real()
    await test_kimi_con_motor_explicito_via_pipeline()
    await test_capability_generate_sin_motor_explicito_resuelve_auto()
    print("E2E: 3/3 OK")


if __name__ == "__main__":
    asyncio.run(main())
