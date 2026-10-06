# C5 auditor SOLO_ORDENES — 2026-10-06

**Tipo:** HISTORIA. **Fuente:** encargo `/home/fruiz/encargos-codex/jax-c5-solo-ordenes.md` y diff/ejecuciones de esta rama. **Decidió:** Fernando Ruiz, 2026-10-06.

Se implementa la opción para permitir auditoría C5 con la faceta de nube configurada en misiones sensibles, solo si `ejecutor.c5_auditor_nube_solo_ordenes` existe como booleano estricto `true`. La proyección HTTP usa lista blanca: objetivo, contrato de auditoría, identidad de máquina y número/comando por paso. No serializa capturas, stderr, líneas, contexto, claims ni campos extra de entrada. Las afirmaciones se conservan para supervisión humana y quedan marcadas como no auditadas por C5. Una respuesta que evalúe afirmaciones o use tipos de hallazgo no permitidos falla cerrada.

La selección registra faceta, proveedor, localidad y modo en la bitácora de misión y el journal de arranque. Para el C3 append-only, el vigía usa un socket Unix autenticado por UID hasta el proceso proxy, que sigue siendo el único escritor. La metadata se agrega a la cadena y, si el canal no está configurado o falla, el modo SOLO_ORDENES no inicia. La revisión final no reenvía claims ni un lote vacío; el vigía audita las órdenes por lotes. Las órdenes Skill/incompletas fallan cerradas porque no caben en la allowlist de comandos. La clave se debe sembrar false por migración en jax-platform y el endpoint de misiones de plataforma debe considerar el modo SOLO_ORDENES; eso requiere PR separado después del PR JAX.

La reauditoría encontró que los turnos sucesivos reutilizan el ID de misión y que C3 rechazaba cualquier repetición. Se corrigió para aceptar idempotentemente la selección idéntica y rechazar una selección conflictiva; la bitácora visible también conserva proveedor y localidad. La segunda revisión detectó que el índice solo vivía en memoria y tenía límite de 4096; ahora se reconstruye al iniciar desde la cadena C3 íntegra, no expulsa misiones y falla cerrado ante selecciones históricas conflictivas. Se cubrió reiniciar el proxy sobre el mismo registro.

Pruebas: baseline de diez archivos contra `origin/master`, 470 recolectadas; rama, 493; delta +23. Se confirmó TDD rojo contra baseline para selección, proyección HTTP, rechazo de Skill/entradas inválidas, schema estricto, separación de instrucciones, modo de canario, código seguro de error HTTP y el rechazo de un auditor local en modo SOLO_ORDENES. No se ejecutaron pruebas DB ni se usó `jax_memory`.

La única llamada real solicitada usó el resolvedor y cliente normal con una misión sintética y 128 tokens máximos. El resolvedor falló cerrado antes de HTTP: `FacetUnavailableError` (su log interno solo registró `RuntimeError`); no fue posible verificar saldo/proveedor y no se repitió la llamada.

Pendiente de cierre: auditoría independiente Tier 3 sobre el SHA corregido y PRs separados de JAX y jax-platform.
