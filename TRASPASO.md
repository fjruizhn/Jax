# Traspaso Codex · Faro F1.1 · 2026-10-09

## Estado actual

- #379 (`7cf27a09e00f6fae036bb98dae862bd7dcc53d8f`) fue aprobado por auditoría Tier 3 y pasó todos los checks del PR. Se integró como `d0c8a68f131849fb298c8cda9b5519e304088e11`. El `post-merge-guard` confirmó la corrida `push` de `tests-puros`, el default y que no hubo carrera.
- #373 conserva los commits de provider, el cierre de subclases hostiles y el contexto de leases compartidos. Se publicó al PR #373 desde `codex/faro-373-post379`; head actual antes de este cambio de handoff: `8cbda4385c03cc135748ec33d27c52fab402d86b`. CI y auditoría Tier 3 están en curso para ese SHA.
- Piso combinado de Identity Foundation Shadow medido con el comando exacto del workflow en Python 3.14.4: `816 passed`, cero skipped. El workflow contiene la lista `test_rule_authority_providers.py` tanto en el paso principal como en el paso del piso; `ci/pisos.json` y `docs/ci/pisos.md` reflejan 816.
- Pruebas locales tras combinar el árbol: 273 tests de providers y ataques, más el paso completo de Identity Shadow (816); JSON/YAML y `git diff --check` correctos.
- #371 está reconstruido limpiamente en `/home/fruiz/wt/jax-faro-f11-paso4-rebuild`, rebasado sobre el #373 posterior a #379. Head local `65214d7a` incluye modelos/store y los dos archivos de cola; mantiene `store.record()` dentro de `leases_de_emision`. Piso combinado medido: 968. Faltan integración de #373, auditoría exact-SHA y CI de #371.
- #381 sobre `4d493b988c2cb116c6236ca868a52aa2da90f413` fue rechazado por auditoría Tier 3: la base vieja puede reintroducir el writer de ratificación eliminado por #379 y el verificador no valida cada fila histórica intermedia del log de checkpoints contra el evento correspondiente. Falta resolver esos dos hallazgos sobre la base nueva y auditar otro SHA.
- Fernando decidió conservar la regla de #381: `OVERLAY_ISSUED` requiere ratificación activa, no revocada, del mismo corpus. No cambiar esa semántica.
- #375 y #378 siguen apilados sobre el evaluador/step4; no reapilar antes de terminar #371.

## Especificación y decisiones

- `docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md` exige leases compartidos en orden `pin → clasificación → STOP` y retención hasta persistencia durable. #373 implementa la frontera `leases_de_emision(...)`; #371 no debe agregar un evaluador nuevo, solo mantener el store actual dentro de esa frontera.
- `TrustedPolicyPin` rechaza subclases de `str` en sus cuatro campos para impedir `__eq__` hostil.
- `FormaLimites` queda alineado al schema vigente: `NINGUNA`, `CANTIDAD` o `MONTO`; una acción obligatoria requiere cantidad o monto, y unidad/moneda deben coincidir.

## Próximos pasos

1. Concluir CI y auditoría exacta de #373; integrar con `verify-integration` y `post-merge-guard` si ambos aprueban.
2. Actualizar la base de #371 a master con #373 integrado, confirmar el piso combinado 968 y el estado del test MariaDB conservado; auditar SHA exacto y completar CI.
3. Corregir los dos BLOCKs de #381 sobre la nueva base, conservar la semántica aprobada de overlay, volver a medir, auditar y pasar CI.
4. Reapilar #375 y #378 sobre #371 final; revisar/auditar/CI individualmente.

No desplegar. No afirmar cierre del Faro hasta integrar #373/#371 y resolver #381, #375 y #378.
