/using-superpowers
/ruflo

# Misión: infraestructura de tests para jax-platform

## Contexto

Auditoría de performance/correctness (2026-07-08, re-verificada 6 veces)
identificó 6 bugs críticos y varios altos en websocket_hub.py, resource_manager,
pipelines.py, state.py y el frontend. Ninguno se toca todavía porque no hay
infraestructura de tests — cero pytest/vitest configurado, cero archivos de
test en todo el repo. Esta misión NO arregla ningún bug — solo construye la
base para poder tocarlos después con cobertura de regresión real.

## 1. Reconocimiento

```bash
cd /home/fruiz/jax-platform
cat backend/requirements.txt
cat frontend/package.json
find . -iname "*test*" -not -path "*/node_modules/*" -not -path "*/.venv/*"
find . -iname "pytest.ini" -o -iname "conftest.py" -o -iname "vitest.config.*"
```

Confirmar qué versión de FastAPI/pydantic usa el backend (para elegir versión
compatible de pytest-asyncio) y qué versión de React/Vite usa el frontend
(para vitest vs jest).

## 2. Backend — pytest

- Instalar `pytest`, `pytest-asyncio`, `httpx` (para TestClient async), en el
  venv existente (`.venv/bin/pip install`).
- `backend/pytest.ini` o sección en `pyproject.toml` con config mínima.
- `backend/tests/conftest.py` con fixtures base: cliente de test con
  override de `get_pool()` apuntando a una DB de test (NO la real
  `jax_memory` — usar una base separada `jax_memory_test` o sqlite in-memory
  si el código lo permite sin reescritura mayor).
- Un primer test real, no un placeholder: cubrir `GET /api/health` (ya
  existe, bajo riesgo) y `POST /api/auth/login` con credenciales inválidas
  (ya lo probamos manualmente ayer — convertirlo en test automatizado).

## 3. Frontend — vitest

- Instalar `vitest`, `@testing-library/react`, `@testing-library/jest-dom`.
- `frontend/vitest.config.js` (puede compartir base con `vite.config.js`
  existente vía `mergeConfig`, no duplicar).
- Un primer test real: `useJaxStore.js` — el store que tocamos ayer para
  sacar el JWT de localStorage. Testear que `restoreSession()` no lee de
  localStorage (regresión directa de lo que arreglamos ayer — si alguien
  reintroduce ese bug sin querer, este test lo atrapa).

## 4. CI local mínima (opcional, preguntar antes)

Si es rápido: un script `scripts/test.sh` que corra ambos (`pytest` +
`npm run test`) en secuencia, útil para correr antes de cada commit. No
armar GitHub Actions ni CI remota en esta misión — fuera de alcance, es
infraestructura nueva que amerita su propia conversación.

## 5. Explícitamente NO hacer

- NO arreglar ningún bug de la auditoría de performance en esta misión.
- NO tocar la DB de producción `jax_memory` para nada relacionado a tests.
- NO instalar frameworks de test adicionales sin preguntar (mantenerse en
  pytest/vitest, el estándar para este stack).

## 6. Gate de prueba

```bash
cd backend && .venv/bin/pytest -v
cd frontend && npm run test
```
Ambos deben correr y pasar (aunque sean solo 2-3 tests cada lado) antes de
comitear.

## 7. Reporte final

Archivos creados/tocados, resultado del gate, decisiones de diseño (DB de
test elegida y por qué, estructura de carpetas de tests), y una lista corta
de "próximos tests obvios a agregar" para cuando ataquemos los bugs reales
(ej: test de aislamiento de conexiones WS por tenant, para el Bug #1).
