/using-superpowers
/ruflo

# Misión: Bug #1 (WS sin filtrar por tenant) + path traversal — con TDD real

## Contexto

Auditoría de performance/correctness (2026-07-08, re-verificada 6 veces,
nunca implementada) identificó dos bugs de severidad de seguridad real
(no solo performance) en jax-platform:

1. **Bug #1 — websocket_hub.py hace broadcast a todo el tenant en vez de
   filtrar por usuario.** Un usuario puede ver eventos/mensajes de otro
   usuario del mismo tenant. `_connections` está keyed solo por `user_id`
   puro (no por `user_id + tenant_id`), lo que además causa que dos
   pestañas del mismo usuario se pisen entre sí (cerrar una rompe la otra
   — relacionado con el multi-tab que probamos ayer en el fix de logout).

2. **Path traversal en `GET /api/command/{task_id}`** — hallazgo nuevo,
   detectado dentro de la auditoría de performance, no la de seguridad
   original del 8 de julio.

Ya existe infraestructura de tests real desde la misión anterior
(`backend/tests/`, `jax_memory_test` en MariaDB, `scripts/test.sh`). Usarla
— no reinventar setup.

## 1. Reconocimiento

```bash
cd /home/fruiz/jax-platform
cat backend/api/websocket_hub.py  # o donde viva _connections real
grep -rn "_connections\|websocket" backend/api/*.py | grep -v test
cat backend/api/command.py  # el endpoint con el path traversal reportado
cat backend/tests/conftest.py  # el fixture setup ya existente, para reusar
```

Confirmar la estructura real de `_connections` (dict, por qué está keyed
solo por `user_id`) y el flujo exacto del path traversal en `command.py`
antes de escribir ningún test o fix.

## 2. Worktree aislado

Igual que la misión anterior — usar la skill de git worktrees, no trabajar
directo sobre `master`.

## 3. TDD real — Bug #1 (aislamiento WS)

1. Escribir el test PRIMERO: dos conexiones WS simuladas, `user_id=A` y
   `user_id=B`, mismo tenant. Publicar un evento dirigido a A. Confirmar
   que el socket de B NO lo recibe. Correr el test — DEBE FALLAR (confirma
   que reproduce el bug real, no un bug inventado).
2. Agregar también el test de multi-tab: dos conexiones con el mismo
   `user_id`, cerrar una, confirmar que la otra sigue viva y recibiendo
   eventos.
3. Recién ahí, el fix: `_connections` debe distinguir conexiones por algo
   más específico que `user_id` puro (ej. `(tenant_id, user_id, connection_id)`
   o estructura equivalente que permita múltiples conexiones del mismo
   usuario sin pisarse, y aislamiento real entre usuarios del mismo tenant).
4. Correr los tests de nuevo — deben pasar ahora.

## 4. TDD real — path traversal en /api/command/{task_id}

1. Test primero: request con `task_id` conteniendo `../` o path absoluto,
   confirmar que el endpoint rechaza (400/403/404, no 200 con contenido
   fuera del directorio esperado). Correr — debe fallar contra el código
   actual.
2. Fix: validar/sanitizar `task_id` antes de usarlo en cualquier construcción
   de path (whitelist de caracteres, o `os.path.realpath` + verificar que
   el resultado sigue dentro del directorio base esperado — el patrón
   estándar, no inventar uno nuevo).
3. Test debe pasar.

## 5. Gate completo

```bash
./scripts/test.sh
```
Debe pasar TODO, no solo los tests nuevos — para confirmar que el fix de
WS no rompió nada de lo que ya andaba (ej. el flujo de chat/pipeline que
depende de broadcast).

## 6. Explícitamente NO hacer

- NO tocar los otros bugs de la lista (resource_manager, Zustand sin
  selectores, índices de DB, etc.) — esta misión es SOLO estos dos.
- NO reiniciar ningún servicio en producción sin confirmación explícita
  de Fernando — el cambio a `_connections` toca el mecanismo de WS que
  usan usuarios reales ahora mismo si están conectados.
- NO mergear a master sin que el gate completo pase y sin reporte final.

## 7. Reporte final

Diff completo de ambos fixes, resultado de cada test (antes/después del
fix, mostrando que reprodujo y luego resolvió), y confirmación explícita
de que el gate completo (no solo los tests nuevos) pasa limpio.
