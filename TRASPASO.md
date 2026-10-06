# Traspaso — Proyectos E2b-1

- **Estado:** bloqueado antes de implementar código funcional. Ronda 2 se
  contrastó con los checkouts desplegados y no coincide con el camino actual.
- **Rama:** `feat/proyectos-e2b1`.
- **Base:** `685579466c25771a4df5d5cfd2c51bc555a21fa6` (`origin/master`).
- **Último commit:** `6b91027e172b4ba6d588134e593eead529d08e6f`.
- **Encargo:** `/home/fruiz/encargos-codex/jax-e2b1-biblioteca.md`.
- **Evidencia:** `docs/historia/2026-10-06-proyectos-e2b1-bloqueo-pipeline.md`,
  sección «Ronda 2 — verificación del camino productivo».
- **Hallazgo:** el checkout productivo de JAX está en `68557946`; el ejecutor
  llama a `/motor/dispatch`, cuyo handler rechaza con 410. No hay llamador de
  producción para `governed_dispatch` ni creación de una ejecución B6 por paso
  de Pipeline. Crear ese contrato excede E2b-1.
- **PR:** #359, abierto; actualizado con la evidencia en el commit indicado.
- **Pruebas/EXPLAIN/carga:** no ejecutados; no se implementó ninguna consulta
  ni integración. Pisos de CI intactos.
- **Pendiente para retomar:** resolver en un diseño separado cómo cada paso de
  Pipeline obtiene una ejecución B6 auténtica y cómo la plataforma liga el
  `project_id` validado a esa autoridad. No reactivar `/motor/dispatch`.
- **Siguiente comprobación:** `git -C /home/fruiz/wt/jax-e2b1 status --short --branch`.
- **Restricciones:** no se tocó `PENDIENTES.md`, jax-platform ni producción.

