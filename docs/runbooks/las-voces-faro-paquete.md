# LAS VOCES — paquete Faro permanente para Qwen local

**Contrato:** LV-001B, decisión de Fernando del 2026-10-03. Esta operación publica un paquete de lectura; no crea ejecuciones ni misiones de Faro. `jaxqwen` continúa igual. El código de este procedimiento vive en `ops/las-voces/`; las fases propias de Faro conservan sus contratos.

## Estado y límites

- La configuración del proyecto `projects/las-voces/.qwen/settings.json` fija `contextWindowSize=131072` y registra el MCP `faro-readonly` con `trust: false`. La aprobación local de ese MCP está ligada a la configuración y a la ruta del proyecto. Qwen directo, sin las cuatro variables `JAX_FARO_*`, deja el MCP desconectado.
- El lanzador versionado `ops/las-voces/qwen-auto` se instala después del merge en `~/.local/bin/qwen-auto`. Exige `/etc/jax/las-voces-faro.env` root:root 0644, analiza cuatro valores como datos y los transmite a Qwen. Es una vía de configuración, no una frontera de autoridad: `fruiz` puede lanzar Qwen por otros medios. La frontera de lectura está en el paquete root-owned verificado por `cargar_paquete` y en la lista fija de herramientas MCP.
- En la comprobación del 2026-10-03, root no tenía acceso autenticado a la API privada ni a Git SSH de `fjruizhn/claude-skills`. Hasta que el operador le proporcione acceso **de solo lectura** por ambos canales, el instalador debe fallar cerrado. No se colocan tokens ni llaves en este repositorio ni en el archivo de entorno.

## Preparación de código confiable, después del merge

La operación root no se ejecuta desde `~/jax` ni desde un worktree propiedad de `fruiz`. Primero, un operador instala un snapshot del **SHA integrado y revisado de JAX** bajo `/srv/faro/jax-tools/<JAX_SHA>`, con todos sus ancestros, archivos Python y directorios de importación root-owned y sin escritura de grupo/otros. Obtiene el OID de `master` por la API oficial de GitHub; desde un espejo Git root-owned de JAX, sin replace refs, coteja que `refs/heads/master` sea ese OID exacto; extrae ese commit a un directorio temporal root-owned y lo renombra al destino solo tras el cotejo. Comprueba que el commit contiene el procedimiento auditado y que `git status` del checkout compartido no intervino en los bytes extraídos. El instalador verifica de nuevo propietario, modo y ausencia de symlinks de su propia ruta de código y de los módulos `jax.faro` que importa. Si no puede afirmar esas condiciones, se detiene.

El operador debe tener acceso de lectura separado al repositorio privado `fjruizhn/claude-skills`: autenticación de `gh api` para consultar el default y su OID, y una llave SSH root de solo lectura para el fetch Git. La API se usa como canal independiente del espejo Git; la llave de GitHub debe estar fijada en `known_hosts` de root. Un error de autenticación, red, default branch, OID o propiedad detiene la operación sin promover el archivo de entorno anterior. Root necesita también acceso de lectura a JAX para formar su snapshot root-owned; ese paso no reutiliza el checkout compartido.

## Instalación inicial y renovación

1. Desde el snapshot root-owned de JAX, ejecutar como root `python3 -I ops/las-voces/faro_paquete_permanente.py`. El operador coteja el OID completo del default oficial `main` con `refs/heads/main` del espejo bare `/srv/faro/claude-skills.git`. El procedimiento rechaza `refs/replace`, fuentes no root-owned y cualquier discrepancia. `-I` impide que `PYTHONPATH` o el directorio de trabajo incorporen módulos Python ajenos al snapshot.
2. El procedimiento construye `/srv/faro/ecosistema/<SHA>` con `ConfigFaro(uid_duenio=0, ref_frescura="refs/heads/main")`, exige `verificar_contra_arbol(cfg) == ()` y carga el paquete antes de publicar `/etc/jax/las-voces-faro.env` por rename atómico. El archivo final es root:root 0644 y contiene únicamente `JAX_FARO_REPO`, `JAX_FARO_SHA`, `JAX_FARO_ECOSISTEMA_DIR` y `JAX_FARO_REF_FRESCURA`. No contiene credenciales ni `JAX_FARO_DUENIO_UID`.
3. Instalar la copia versionada del lanzador en `~/.local/bin/qwen-auto` con propietario `fruiz` y modo 0755. Registrar el SHA de JAX usado para esa copia y contrastarla byte por byte con `ops/las-voces/qwen-auto` del snapshot revisado. Esto es despliegue y se hace únicamente tras el merge y el GO correspondiente.
4. Para renovar al cambiar `claude-skills/main`, repetir el paso 1 desde un snapshot JAX root-owned. Solo un paquete nuevo verificado provoca el rename del env. Se conservan los directorios de SHA anteriores para sesiones vivas; no se borran durante la renovación. Si falla cualquier paso, el env previo queda vigente y se reporta el atraso.

El valor que debe cargar una sesión nueva es el SHA exacto de `/etc/jax/las-voces-faro.env`, no el `main` que pueda avanzar después. La frescura se reporta aparte de la integridad.

## Aceptación diferida de LV-001

Tras el despliegue y **cuando termine la reserva de GPU de la misión 4b**, desde `~/jax/projects/las-voces` se inicia una sesión normal mediante el lanzador. Se registra tarea, SHA del paquete, SHA de JAX, versión Qwen Code, contexto efectivo 131072, MCP `Connected` y las llamadas reales con sus resultados a `skills.buscar`, `skills.leer` y `agentes.listar`. Se comprueba que la interfaz no anuncia operaciones de crear/lanzar/mutar, y que `jaxqwen` permanece intacto. El smoke aislado anterior de PR #327 no sustituye esta prueba de la sesión normal.

La aprobación MCP que Fernando dio en `~/jax/projects/las-voces` estaba ligada a la configuración vigente. Si al desplegar o cambiar la ruta/configuración Qwen vuelve a pedir confianza, Fernando aprueba de nuevo `faro-readonly` allí antes del smoke. Los resultados se ligan a LV-001 y al SHA exacto; Fernando acepta conforme a LV-003. Hasta entonces el criterio de sesión normal y la aceptación humana siguen abiertos; no se marca LV-001 como `DONE`.
