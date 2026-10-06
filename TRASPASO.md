# Traspaso — Proyectos E2b-1

- **Estado:** implementación funcional detenida por incompatibilidad confirmada
  entre el spec cerrado y el contrato actual de Pipeline; reporte listo para PR.
- **Rama:** `feat/proyectos-e2b1`.
- **Base:** `685579466c25771a4df5d5cfd2c51bc555a21fa6` (`origin/master`).
- **Encargo:** `/home/fruiz/encargos-codex/jax-e2b1-biblioteca.md`.
- **Hallazgo:** `/motor/dispatch` responde `410 GOVERNED_EXECUTION_REQUIRED`;
  `/motor/governed-dispatch` no conserva `pipeline_id` ni `project_id`; Pipeline
  no persiste `project_id`. Evidencia ampliada en
  `docs/historia/2026-10-06-proyectos-e2b1-bloqueo-pipeline.md`.
- **Auditoría:** escalón 3, solo lectura, veredicto de incompatibilidad en este
  SHA; no consultó DB ni producción.
- **Cambios:** solo registro del hallazgo y este traspaso. Sin código funcional,
  `PENDIENTES.md`, CI, base de datos o producción.
- **Pruebas/EXPLAIN/carga:** no ejecutados; no hay query ni herramienta integrada
  que medir. Pisos CI intactos.
- **Siguiente paso:** publicar PR de documentación que presenta el bloqueo.
  Retomar implementación después de resolver el contrato de autoridad/proyecto
  en un diseño separado.
