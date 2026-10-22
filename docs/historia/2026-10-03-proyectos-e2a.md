# Proyectos E2a — 2026-10-03

> HISTORIA. Autor: Mr. Hyde (hall9000, sesión fruiz-12). Decisiones: Fernando, en persona y vía jax-14.
> Spec: `docs/superpowers/specs/2026-10-02-proyectos-e2-adenda-design.md` · Plan: `docs/superpowers/plans/2026-10-03-proyectos-e2a-plan.md`
> Runbook: `docs/runbooks/proyectos-e2a-produccion.md` · Registro del despliegue: `~/respaldos-despliegue/2026-10-03-e2a/` en hall9000 (CHECKLIST.md, fotos, respaldos, mapa de reversión de LACTOVI).

## Qué se hizo
- **Decisiones de Fernando:**
  - (§8 de la adenda) Ocultar un documento entra en E2a. El Ejecutor (E2b-3) queda fuera de E2.
  - El plan se ejecuta con el GO «dale con todo».
  - El escáner de política excluye las carpetas ocultas.
- **Parte A (jax), Jax#325, merge `2de1d1e`.**
  - Migración 006a (`project_documents`).
  - LAS MANOS procesa por `project_uuid`: 422 `project_uuid_invalido` y `proyecto_no_activo`, 503 `base_no_disponible`.
  - La ingesta procesa en el lugar lo que ya vive en `fuente/` (`_origen_ya_en_fuente`).
  - `ops/permisos_proyectos.py`: dueño jaxsvc, grupo fruiz, ACL y setgid, núcleo privilegiado instalado por root.
  - `scripts/proyectos_e2a_lactovi.py`: ensayo, `--aplicar`, `--completar` y `--revertir`.
  - El runbook.
  - Jax#328 corrigió la regla de sudoers: el `^` va escapado para sudo 1.9.17. Lo detectó jax-14 al instalarla.
- **Parte B (jax-platform), jax-platform#182, merge `ca83996`.**
  - API `/api/proyectos/{id}/documentos` para subir, listar, ocultar y restaurar.
  - Despachador de fondo hacia LAS MANOS.
  - Pestaña Documentos y selector con resumen antes de subir.
  - Botón 📄 en la barra del chat.
  - Topes en `axioma_config`.
  - La rama se resolvió contra master después de #175 (F2-E-SR, Codex). Hubo una auditoría corta de escalón 3 sobre `5ffe658`: APROBADO, sin hallazgos. Los pisos de CI se fijaron con el número del runner (con DB 3387, sin DB 2064, frontend 1332), sin bajar ninguno.
- **Parte C (producción), hoy.** Va en la sección siguiente.

## Despliegue (Parte C)
- **GO.** Fernando, vía jax-14. Primero se pospuso: opción B, después de la misión 4b del Ejecutor y de la prueba de carga de F2-E-SR. Después se adelantó: opción A, ~11:10, con la misión 4b en espera y en un solo despliegue junto con el «pensamiento apagado» del proxy (Jax#332, d334e8a).
- **Qué se desplegó.**
  - jax pasó de `eca7db42d` a `c8b05a4`, e incluye #313 F2-E-SR, #320/#327 Las Voces, #322, #326, #328, #329 y #332.
  - jax-platform pasó de `9a2c90c2` a `610a084`, e incluye #175 F2-E-SR (en par exacto con jax#313), #182 y #183.
  - Sitio público: `index-B2DmWcwL.js` pasó a `index-DcJmrnIe.js`.
- **Respaldo de `jax_memory` con restauración probada.** A las 06:38 y de nuevo fresco a las 10:15. Las dos veces, 10 de 10 conteos, 72 de 72 tablas y 5 de 5 triggers iguales.
- **Condiciones del ensayo 4b (jax-3a), respetadas.**
  - Antes y después se sacó una foto, y quedaron iguales: `politica.json`, `cerco.nft`, la tabla `inet ejecutor_cerco`, el `+a` del registro y todas las `JAX_EJECUTOR_*` y `JAX_PROXY_CARRIL_*`.
  - En `/etc/jax/.env` se agregó una sola línea: `JAX_PROXY_CARRIL_PENSAMIENTO=apagado`.
  - El proxy y el freno los reinició jax-3a. La misión negativa se rechazó solo por `maquina_sin_contratos_remotos` y no hizo falta el instalador.
- **nginx de axioma-ia.io (atem-ai).**
  - `/api` heredaba `client_max_body_size 50m`. Se agregó `location ~ ^/api/proyectos/[^/]+/documentos$` con 1100m, `proxy_request_buffering off`, el mismo `limit_req` de `/api/chat/upload` y timeouts de 300 s.
  - **La autenticación va antes del cuerpo**, medido desde afuera: un POST de 200 MB sin token dio 401 en 0,04 s, con 128 KB subidos y nada en el TMPDIR (`/srv/jax-data/tmp`).
- **Verificaciones.**
  - LAS MANOS responde 422 `proyecto_no_activo` con un uuid inexistente.
  - `project_documents` tiene `uq_project_documents_sha` y las 3 FK.
  - Las 6 claves `proyectos.documentos.*` quedaron sembradas.
  - `/api/proyectos/documentos/limites` sin token da 401.
- **Permisos de `proyectos/`.** `--verificar` daba 378 NO CUMPLE. Se corrió `--aplicar` (111 directorios, 267 archivos) y `--verificar` quedó en 0. La prueba real de escritura cruzada jaxsvc↔fruiz pasó en las dos direcciones, aun con `umask 077`.
- **LACTOVI, de carpeta suelta a proyecto 3** (uuid `01a1029d-3078-7707-aab7-2000d235137f`).
  - Ensayo: 120 archivos, 88 fichas, 18 duplicados por sha y 102 filas.
  - 5 fichas sin archivo: 4 son extractos de versiones viejas de archivos reemplazados el 26-sep, y 1 es `MONTECARLO_DSCR.png`, que ya no está en `fuente/`. Esas carpetas viajan con el rename y no se registran.
  - Respaldo restic previo: local `501e678f` y R2 `7899da34`.
  - Después de `--aplicar`, el despacho de las 11 filas en cola terminó con listo 24, parcial 32, error 30 y sin_extractor 16. `fuente/` quedó en 120 archivos con el mismo hash (`7b8f3868…`): no hubo copias.
  - Restauración probada posterior (paso 7): restic local `d5982839` y R2 `6ede2942`. Restaurado desde los dos: 120 archivos y los 102 sha256 iguales a `project_documents`.
- **Segundo despliegue (SR3 + auditor C5), 11:55.** jax `0166f7e`, plataforma `1e2888c`, `ejecutor.c5_tope_s=400`.

## Tercer despliegue (2026-10-04, 04:04 a 04:20)
- **GO.** Fernando, transmitido por la coordinación (jax-14, después jax-fe). La ventana se verificó con `bin/ventana estado` antes de cada paso irreversible, y la pausa C5 estaba puesta.
- **Qué entró.** jax `0166f7e` → `5f4f5e0`:
  - #337: el workspace sin ruta por defecto, que falla cerrado;
  - #338: una imagen sin texto es un documento válido, con reglas A, B, C y D, doble pasada blanco/negro con unión para la transparencia, MPO cuadro por cuadro y `formato_no_soportado:<f>`;
  - #340: los permisos sin acceso de «otros», con la raíz del workspace en 770;
  - #343: los pisos de CI en `ci/pisos.json`, con el job aislado `pisos-no-bajan`.

  Plataforma `1e2888c` → `103755d`: #186 («Reprocesar») y #187 (mensajes por formato no soportado).
- **Pasos.**
  - respaldo con restauración probada (72/72 tablas y 5/5 triggers; hay que quitar `DEFINER` al restaurar);
  - foto previa;
  - stop de la plataforma → jax → LAS MANOS → plataforma (con la migración de las claves `reprocesar_*`) → frontend → sitio (`index-BfWY7jhC.js`);
  - `limit_req` de nginx en `…/reprocesar`;
  - reinstalación del núcleo de permisos;
  - bloque `--aplicar` del runbook: 126 directorios y 290 archivos, solo `other::---`;
  - quitar el symlink `/home/fruiz/jax-workspace`;
  - foto posterior: igual en todo lo del Ejecutor.
- **Dos desviaciones del bloque del runbook**, corregidas en Jax#345:
  1. El sshfs `/home/fruiz/atem-ai` da EACCES incluso a root, y la premisa de setuid cortaba. Se toleró solo ese error en un punto de montaje FUSE.
  2. `jax-ejecutor-proxy` no volvió por un socket viejo (`/run/jaxqwen-proxy/jaxqwen-model.sock`), y el trap lo dio por restaurado porque lo vio activo un instante. jax-3a lo levantó. El trap ahora exige `active` estable y sin reinicios.
- La carpeta oculta `.claude-flow` (estado de Ruflo) frenaba `--aplicar`. La borró Fernando, como pide el runbook.

## Prueba de humo (Fernando, 2026-10-04 04:31; verificada por Hyde)
- Se reprocesaron 28 JPEG y 4 PDF, y ninguna imagen queda en error (antes había ~30). En las fichas:
  - 11 `imagen_sin_texto`, en listo;
  - 16 `imagen_texto_dudoso`, en parcial;
  - 1 `imagen_pagina_sin_texto`, en parcial: el JPEG de un RTN.
- 27 fichas se leyeron en 2 cuadros, porque 24 de las 41 fotos son MPO de iPhone. En 5 apareció texto del cuadro 2.
- `fuente/` sin cambios: 123 archivos con el mismo hash de contenido.
- «DUI RMR.pdf» queda en `error/ocr_sin_texto`, que es lo esperado. Por la decisión del 10-03, los PDF escaneados sin texto siguen en error; la regla D es solo para imágenes. Son dos fotos de 78–82 ppi dentro de un PDF exportado de Word, y hace falta un escaneo mejor.

## Incidente y corrección (Principio VIII)
- **La diferencia en la restauración del respaldo previo.** Difirieron 2 archivos de `.claude-flow/` (estado de Ruflo), reescritos 32 s después del snapshot. Los 267 archivos del cliente salieron iguales. El runbook pide escalar ante cualquier diferencia, y se escaló. **Fernando aceptó seguir** (~11:45, vía jax-14).
- **`fuente/` marcó 122 durante el despacho.** Los 2 archivos de más eran otro `.claude-flow`, creado por el gancho de Ruflo de la propia sesión de Hyde. Su directorio de trabajo había quedado dentro de `fuente/` tras un `cd` de verificación, y ese gancho escribe estado en el directorio de trabajo. Se sacó la sesión de ahí, se quitó la carpeta (solo 2 archivos de herramienta) y `fuente/` volvió a 120 con el hash idéntico.
- **Corrección dicha a Fernando:** la diferencia de la restauración también la había causado ese gancho, no «Ruflo por su cuenta». La decisión sigue valiendo sobre los hechos.
- **Lección:** ninguna sesión deja su directorio de trabajo dentro de una carpeta de datos de un cliente. Se lee con rutas absolutas.

## Lecciones técnicas
- **sudoers 1.9.17:** un argumento que empieza con `^` es una regex. Para que sea literal, va como `\^`.
- **Ganchos y comprobaciones de seguridad:**
  - El freno de seguridad de Claude Code bloquea `trap "rm -rf $T"` dentro de `sudo bash -c`. Sin archivo temporal, con la contraseña en `MYSQL_PWD` dentro del proceso root, el respaldo pasa sin esquivar nada.
  - Los ganchos de las herramientas escriben en el directorio de trabajo, ver el incidente.
- **El checkout de producción de jax es de jaxsvc.** Se trae con la llave de fruiz prestada, y después se devuelve el dueño solo de lo que quedó en root (`find -user root -exec chown -h`), no con `chown -R` sobre todo el árbol. Quedaron 87 archivos en root y se corrigieron a 0.
- **`--verificar` no miraba los bits de «otros».** Lo corrigió Jax#340.
- **Un traslado invalida los argumentos de seguridad que dependen de la ruta.** Al mover el workspace a `/srv`, desapareció la barrera de `/home/fruiz` (750) y cualquier usuario local podía leer los documentos. jax-14 lo cerró con `chmod 770` en la raíz, y la causa de fondo la resolvió #340. Desde entonces, después de cualquier traslado se vuelve a medir con una lectura real (`sudo -u nobody head -c1`), no con `test -r`: el `test` de uutils ignora las ACL.
- **GitHub deja de ejecutar un workflow de más de unos 512 000 bytes**, y lo marca como «workflow file issue». Cinco rondas de #338 no corrieron el CI principal y no se notó, porque los checks que sí aparecían estaban en verde. Lo arregló #343: los pisos pasaron a datos. Lección: comprobar que corrió cada workflow esperado, no solo que lo visible esté en verde.
- **Revisar el CI en cada ronda, no solo al final.** #340 estuvo varias rondas en rojo en el runner, porque las pruebas corren como un usuario que no es fruiz.
- **Un encargo mal formulado puede autorizar algo prohibido.** Pedir «un usuario efímero con sudo» llevó a tocar `/etc/sudoers.d`, aunque se borró enseguida. Se avisó a Fernando.
- **Cuando cada arreglo trae un defecto nuevo, simplificar.** Elegir el fondo por luminancia, después elegir entre dos pasadas y después deduplicar con dos fuentes abrieron un defecto por ronda. Se cerró con dos invariantes: con transparencia real nunca sale `ok`, y nunca se pierde una línea de texto.

## Alternativas descartadas
- **Bajar los pisos de CI tras la fusión con #175.** La fórmula daba 3262 y 1912. Descartado: se fijan con el número real del runner.
- **Desplegar un subconjunto de master armado a mano** (solo E2a). Descartado: se despliega master completo con el acuerdo de las sesiones dueñas.
- **Comparar la restauración excluyendo `.claude-flow` para que dé «iguales».** Descartado: era mover el criterio. Se escaló.

## Pendientes
- ~~`proyectos/` conserva `other::r-x`~~: **resuelto** por Jax#340 y aplicado en el tercer despliegue.
- **Para que Fernando decida:** si los PDF escaneados sin texto deberían tratarse como las imágenes (parcial con código) en lugar de quedar en error.
- **Para que Fernando decida:** volver a fijar los `2 skipped` del piso de OCR. Obliga a cambiar la forma del patrón, y eso solo lo permite tocando `.github/ci`.
- Jax#345 (arreglos del bloque del runbook): en auditoría.
- **Para que Fernando decida** (ya escrito en el runbook): el modelo jaxsvc-dueño permite que un proceso jaxsvc reescriba la ACL. Entra en el tema de identidades del 06-oct.
- **DECISIÓN (Fernando, 2026-10-03 ~13:05):** jaxsvc NO entra al grupo fruiz («fruiz tiene acceso ilimitado, no es conveniente»). El setgid que pierde `almacen.py` (un `fchmod` de jaxsvc, que no es miembro del grupo, borra S_ISGID) se resuelve solo en código, en jax-platform#186: no chmod de carpetas, verificar la herencia y fallar cerrado. Subtema cerrado.
- La rotación del token del bot de avisos de Telegram va después de la misión 4b (jax-14 con Fernando).
- E2b: queda para la siguiente entrega.
