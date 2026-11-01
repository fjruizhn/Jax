# Node del ejecutor
## Purpose
Documentar el árbol `/opt/ejecutor/node-v24.22.1` que usa el ejecutor (Claude Code como usuario `axioma`) y cómo reconstruirlo.
## Scope
`/opt/ejecutor/node-v*`, `JAX_EJECUTOR_NODE_BIN` y `scripts/ejecutor_fase0/harness.py`. Hecho el 2026-11-01 con GO de Fernando.
## Preconditions
- GO de Fernando o ventana abierta: se escribe en `/opt` como root.
- Estado actual: `/opt/ejecutor/node-v24.22.1` es el tarball oficial de nodejs.org (SHA256 `6c172a9ab6718f7b801fb5a7bd486af2ee5e32758086a04b84fcce2aa77095e6`), extraído como root.
- Dentro van `lib/node_modules/@anthropic-ai/claude-code` fijado en 2.1.273 y `ruflo` 3.36.0, copiados del árbol anterior, con los enlaces `bin/claude` y `bin/ruflo`. Propiedad `root:root`, permisos `a+rX,go-w`.
## Authority impact
Ninguno sobre `policy/**`. Cambia el binario que corre el ejecutor; la ruta vive en `JAX_EJECUTOR_NODE_BIN`, en `/etc/jax/.env` y `/etc/jax/runtime.env`.
## Safe procedure
1. Verificar el SHA256 del tarball contra el publicado por nodejs.org antes de extraer; extraer como root en `/opt/ejecutor/node-v<versión>`.
2. Instalar las versiones fijadas (claude-code 2.1.273, ruflo 3.36.0) con npm 11.19 usando `--ignore-scripts --allow-git=none`; ese npm rechaza dependencias git.
3. Crear los enlaces `bin/claude` y `bin/ruflo`; dejar `root:root` y `chmod -R a+rX,go-w`.
4. Actualizar `JAX_EJECUTOR_NODE_BIN` en `/etc/jax/.env` y `/etc/jax/runtime.env`, y `NODE_BIN` en `scripts/ejecutor_fase0/harness.py`. Solo `jax-platform` necesita reinicio.
5. `/opt/ejecutor/node-v*` NO se respalda (excluido en `backup-hall9000.sh`): tras restaurar hay que reconstruirlo con los pasos 1 a 3.
## Verification
Correr `scripts/ejecutor_contratos/probar_arranque.py` como `jaxsvc`, sin imprimir secretos. Ojo: consume el modelo local y deja la VRAM cargada una hora.
## Fail-closed condition
Si falta `bin/claude` o el SHA256 no coincide, no se apunta `JAX_EJECUTOR_NODE_BIN` al árbol nuevo.
## Recovery / escalation
La reversión exige reconstruir el árbol anterior (ya no existe en el host). Escalar a Fernando.
## Prohibited actions
No instalar Node vía NodeSource. No correr scripts de instalación de npm (`--ignore-scripts`). No dejar el árbol escribible por grupo u otros.
