# Node del ejecutor
## Purpose
Documentar el árbol `/opt/ejecutor/node-v24.22.1` que usa el ejecutor (Claude Code como usuario `axioma`) y cómo reconstruirlo.
## Scope
`/opt/ejecutor/node-v*`, `JAX_EJECUTOR_NODE_BIN` y `scripts/ejecutor_fase0/harness.py`. Hecho el 2026-11-01 con GO de Fernando.
## Preconditions
- GO de Fernando o ventana abierta: se escribe en `/opt` como root.
- Estado actual: `/opt/ejecutor/node-v24.22.1` es el tarball oficial de nodejs.org (SHA256 `6c172a9ab6718f7b801fb5a7bd486af2ee5e32758086a04b84fcce2aa77095e6`, publicado en `SHASUMS256.txt` de esa versión), extraído como root.
- Dentro van `lib/node_modules/@anthropic-ai/claude-code` fijado en 2.1.273 y `ruflo` 3.36.0, copiados del árbol anterior, con los enlaces `bin/claude` y `bin/ruflo`. Propiedad `root:root`, permisos `a+rX,go-w`.
## Authority impact
Ninguno sobre `policy/**`. Cambia el binario que corre el ejecutor; la ruta vive en `JAX_EJECUTOR_NODE_BIN`, en `/etc/jax/.env` y `/etc/jax/runtime.env`.
## Safe procedure
1. Comprobar el SHA256 del tarball contra `SHASUMS256.txt` de nodejs.org antes de extraer; extraer como root en `/opt/ejecutor/node-v<versión>`.
2. Instalar las versiones fijadas (claude-code 2.1.273, ruflo 3.36.0) con `npm install -g --prefix <arbol> --ignore-scripts --allow-git=none <paquete>@<versión>`. `--allow-git=none` es lo que hace que npm rechace dependencias git; no es el comportamiento por defecto.
3. `--ignore-scripts` deja `lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe` como un stub de unos 500 bytes: el postinstall que lo reemplaza no corre. Hay que colocar a mano el binario nativo del paquete `claude-code-linux-x64`, que npm sí instala como dependencia. En el árbol actual está en `<arbol>/lib/node_modules/@anthropic-ai/claude-code/node_modules/@anthropic-ai/claude-code-linux-x64/claude`. Como enlace duro (mismo sistema de archivos) o copia:
   ```
   PKG=<arbol>/lib/node_modules/@anthropic-ai/claude-code
   ln -f $PKG/node_modules/@anthropic-ai/claude-code-linux-x64/claude $PKG/bin/claude.exe
   # o, si no se puede enlazar:
   install -m 0755 $PKG/node_modules/@anthropic-ai/claude-code-linux-x64/claude $PKG/bin/claude.exe
   ```
   En el árbol actual los dos archivos son el mismo (enlace duro, 228663608 bytes).
4. Crear los enlaces `bin/claude` y `bin/ruflo`; dejar `root:root` y `chmod -R a+rX,go-w <arbol>`.
5. Actualizar `JAX_EJECUTOR_NODE_BIN` en `/etc/jax/.env` y `/etc/jax/runtime.env`, y `NODE_BIN` en `scripts/ejecutor_fase0/harness.py`. Solo `jax-platform` necesita reinicio.
6. `/opt/ejecutor/node-v*` NO se respalda (excluido en `backup-hall9000.sh`): tras restaurar hay que reconstruirlo con los pasos 1 a 4.
## Verification
- `<arbol>/bin/claude --version` debe imprimir `2.1.273` y `<arbol>/bin/node -v` debe imprimir `v24.22.1`.
- Correr `scripts/ejecutor_contratos/probar_arranque.py` como `jaxsvc`, cargando el entorno como indica la cabecera del script y sin imprimir secretos. Ojo: al exportar C1 reescribe `/etc/jax-ejecutor/politica.json`. Carga el modelo local en la GPU; se descarga con `keep_alive` 0.
## Fail-closed condition
No se apunta `JAX_EJECUTOR_NODE_BIN` al árbol nuevo si el SHA256 no coincide, si `bin/claude --version` no da exactamente 2.1.273 o si `bin/node -v` no da exactamente v24.22.1. Que el archivo exista no basta.
## Recovery / escalation
La reversión exige reconstruir el árbol anterior (ya no existe en el host). Escalar a Fernando.
## Prohibited actions
No instalar Node vía NodeSource. La prohibición de correr scripts de npm sigue vigente (`--ignore-scripts`): por eso el binario nativo se coloca a mano (paso 3) y no con el postinstall. No dejar el árbol escribible por grupo u otros.
