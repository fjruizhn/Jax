# C5 auditor SOLO_ORDENES — 2026-10-06

**Tipo:** HISTORIA. **Fuente:** encargo `/home/fruiz/encargos-codex/jax-c5-solo-ordenes.md` y diff/ejecuciones de esta rama. **Decidió:** Fernando Ruiz, 2026-10-06.

Se implementa la opción para permitir auditoría C5 con la faceta de nube configurada en misiones sensibles, solo si `ejecutor.c5_auditor_nube_solo_ordenes` existe como booleano estricto `true`. La proyección HTTP usa lista blanca: objetivo, contrato de auditoría, identidad de máquina y número/comando por paso. No serializa capturas, stderr, líneas, contexto, claims ni campos extra de entrada. Las afirmaciones se conservan para supervisión humana y quedan marcadas como no auditadas por C5. Una respuesta que evalúe afirmaciones o use tipos de hallazgo no permitidos falla cerrada.

La selección registra faceta y modo en eventos de misión, y el arranque emite el dato al journal. El C3 append-only lo escribe el proceso proxy y no comparte escritor con arranque; no se agregó un escritor paralelo. La clave se debe sembrar false por migración en jax-platform y el endpoint de misiones de plataforma debe considerar el modo SOLO_ORDENES; eso requiere PR separado después del PR JAX.

Pruebas: baseline original de nueve archivos, 447 recolectadas; rama, 462; delta +15. Focal: 462 passed. Se confirmó TDD rojo contra baseline para selección, proyección HTTP, modo de canario, y código seguro de error HTTP. No se ejecutaron pruebas DB ni se usó `jax_memory`.

Pendiente de cierre: medición completa de `tests-puros/out` y actualización del piso sobre la base actualizada, llamada real mínima por el resolvedor/cliente `thot`, auditoría independiente Tier 3 y PRs separados de JAX y jax-platform.
