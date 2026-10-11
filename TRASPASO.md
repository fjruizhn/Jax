# Traspaso · Rule Authority checkpoint durability

- Rama: `fix/faro-checkpoint-restart-recovery`.
- Base/head local verificado al retomar: `d0991f183f1cb9559684147b4d83f11e2e159e3b`; trabajo local aún sin commit.
- Objetivo: cerrar los hallazgos Tier 3 del SHA `d0991f18` y la dirección de Fernando: el estado `DURABILITY_UNKNOWN` debe bloquear retries hasta probar durabilidad por fsync y relectura validada; cualquier imposibilidad niega y conserva en DB las claves de evaluación/consumo.
- Cambios en curso: estado de checkpoint desconocido por instancia (incluye cada instancia nueva), reconciliación `fsync` de archivo y cadena de directorios, identidad `(st_dev, st_ino)` estable y hash-chain exacta; traducción de toda ruta `head_actual()`/`confirmar()` a error de servicio estable; tipos públicos separados para falla genérica post-commit y durabilidad desconocida. La prueba unitaria ahora cubre también la traducción del error en ambas fronteras y verifica que una falla genérica no se etiqueta `DURABILITY_UNKNOWN`.
- Pruebas: comando exacto del workflow local dio `58 passed` y `piso.py verificar authority-rule-unit/tests` pasó; colección MariaDB dio 47. Los tests integrados de retry/restart y crash fueron intentados pero Docker negó acceso a `/var/run/docker.sock`; requieren CI.
- Siguiente comando: `pytest -q tests/policy/test_rule_authority_checkpoint.py tests/policy/test_faro_rule_authority_store.py tests/policy/test_faro_rule_permit.py tests/policy/test_faro_rule_authority_kernel.py`.
- Falta: revisar el diff completo, compilar, medir pisos exactos, actualizar esta historia, commit local, eliminar este traspaso antes de invocar auditoría Tier 3 exacta. Después: solicitar auditoría independiente, push/CI solo tras veredicto, y verificar procedimiento/ventana antes de integración.
- No publicar ni integrar este SHA sin nueva auditoría exacta; no se afirma que MariaDB pasó localmente.
