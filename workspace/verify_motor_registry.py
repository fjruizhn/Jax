#!/usr/bin/env python3
"""Verificación del Motor Registry v0.1 skeleton."""
import sys
import tomllib

sys.path.insert(0, "/home/fruiz/jax/las_manos")

print("=== py_compile ===")
import py_compile
files = [
    "/home/fruiz/jax/las_manos/motor_registry/__init__.py",
    "/home/fruiz/jax/las_manos/motor_registry/models.py",
    "/home/fruiz/jax/las_manos/motor_registry/job_store.py",
    "/home/fruiz/jax/las_manos/motor_registry/catalog.py",
    "/home/fruiz/jax/las_manos/motor_registry/policy.py",
    "/home/fruiz/jax/las_manos/motor_registry/routes.py",
    "/home/fruiz/jax/las_manos/motor_registry/worker.py",
]
all_ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"  OK  {f.split('/')[-1]}")
    except py_compile.PyCompileError as e:
        print(f"  FALLO  {e}")
        all_ok = False

print()
print("=== import models ===")
from motor_registry.models import JobStatus, MotorDispatchRequest, MotorDispatchResponse, MotorJobView
print(f"  JobStatus values: {[s.value for s in JobStatus]}")
print(f"  MotorDispatchRequest fields: {list(MotorDispatchRequest.model_fields.keys())}")
print("  OK")

print()
print("=== catalog + policy: rechaza caller no autorizado ===")
with open("/home/fruiz/jax/las_manos/config.toml", "rb") as f:
    config = tomllib.load(f)

from motor_registry.catalog import MotorCatalog
from motor_registry.policy import MotorPolicy

catalog = MotorCatalog(config)
policy = MotorPolicy(catalog)

# jekyll NO está en allowed_callers de code_swarm
r1 = policy.check(
    caller="jekyll",
    capability="code_swarm",
    motor=None,
    context_keys=[],
    recursion_depth=0,
    human_gate_token=None,
)
print(f"  jekyll → code_swarm: allowed={r1.allowed}, reason='{r1.reason}'")
assert not r1.allowed, "FALLO: debería rechazar jekyll"

# hyde SÍ está en allowed_callers de refactor, sin human gate
r2 = policy.check(
    caller="hyde",
    capability="refactor",
    motor=None,
    context_keys=[],
    recursion_depth=0,
    human_gate_token=None,
)
print(f"  hyde → refactor: allowed={r2.allowed}, motor='{r2.resolved_motor}', reason='{r2.reason}'")
assert r2.allowed, "FALLO: debería permitir hyde"

# hyde → code_swarm SIN token: rechaza por human gate
r3 = policy.check(
    caller="hyde",
    capability="code_swarm",
    motor=None,
    context_keys=[],
    recursion_depth=0,
    human_gate_token=None,
)
print(f"  hyde → code_swarm sin token: allowed={r3.allowed}, reason='{r3.reason}'")
assert not r3.allowed, "FALLO: debería rechazar sin token"

# hyde → code_swarm CON token: pasa
r4 = policy.check(
    caller="hyde",
    capability="code_swarm",
    motor=None,
    context_keys=[],
    recursion_depth=0,
    human_gate_token="tok_test_abc123",
)
print(f"  hyde → code_swarm con token: allowed={r4.allowed}, motor='{r4.resolved_motor}'")
assert r4.allowed, "FALLO: debería permitir con token"

# context con clave prohibida
r5 = policy.check(
    caller="hyde",
    capability="refactor",
    motor=None,
    context_keys=["api_key"],
    recursion_depth=0,
    human_gate_token=None,
)
print(f"  hyde → refactor con api_key: allowed={r5.allowed}, reason='{r5.reason}'")
assert not r5.allowed, "FALLO: debería rechazar api_key en context"

print()
print("=== motores en catálogo ===")
print(f"  Motores habilitados: {catalog.enabled_motors()}")
cap = catalog.get_capability("code_swarm")
print(f"  code_swarm: callers={cap.allowed_callers}, motors={cap.allowed_motors}, human_gate={cap.requires_human_gate}")

print()
if all_ok:
    print("VERIFICACIÓN COMPLETA — Motor Registry v0.1 skeleton OK")
else:
    print("VERIFICACIÓN CON FALLOS — revisar arriba")
    sys.exit(1)
