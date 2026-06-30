# Resultado de: jacobs-sizefix.md

---

## Resumen de lo ejecutado

| Item | Antes | Después | Estado |
|------|-------|---------|--------|
| `SIZE_LIMIT` en `artifacts.py` | `1 MB (1_048_576)` | `60_000 bytes (60 KB)` | ✅ en disco |
| `output_ref` en `jacobs_steps` | `TEXT` (~64 KB límite) | `LONGTEXT` (~4 GB límite) | ✅ aplicado en DB |
| Servicio reiniciado | — | pendiente | ⚠️ requiere `sudo` |

**Rollback disponible:** `~/jax/jacobs/artifacts.py.backup-sizefix-20260629-063040`

**Próximo paso:** `sudo systemctl restart jax-las-manos` → luego corro el gate automáticamente. Con Fix A activo, el ensamble de 139 KB irá a `artifact://jacobs/...` y solo un ref corto toca la columna.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
