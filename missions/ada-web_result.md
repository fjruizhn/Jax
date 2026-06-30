# Resultado de: ada-web.md

---

## Reporte final — Misión Ada-Web ✓

**Gate superado.** Respuesta real de Ada via `/api/chat`:
```
"Soy Ada, séptima faceta del concilio JAX/Axioma: transformo la intención
en algoritmos precisos y distingo lo demostrado de lo supuesto."
```

### Criterios de aceptación

| # | Criterio | Estado |
|---|---|---|
| A | Placeholder de Ada eliminado de `chat.py` | ✓ |
| B | Rama `ada` en `_invoke_facet` (OpenAI-compat, glm-5.2, ZAI_API_KEY) | ✓ |
| C | `keys.py` coherente (glm-5.2 / ZAI_API_KEY / test_url real) | ✓ |
| D | `dashboard.py` usa `ZAI_API_KEY` | ✓ |
| E | `usage.py` con precio real glm-5.2 ($1.40/$4.40 per 1M) | ✓ |
| F | `descAda` = analista en ES y EN | ✓ |
| F | `capability: 'analysis'` en PipelineModal | ✓ |
| G | Ada responde por la web con glm-5.2 | ✓ (HTTP 200, respuesta real) |
| H | Fila en MariaDB: `role='ada'`, `model='glm-5.2'` | ✓ (id=711, 04:46:44) |
| I | Panel admin: test_url configurado para Z.ai | ✓ (ahora apunta a `/api/paas/v4/models`) |

### Archivos modificados
- `backend/api/chat.py` — elimina placeholder líneas 324-327; agrega rama `ada` en `_invoke_facet`
- `backend/api/admin/keys.py` — glm-5.2 / ZAI_API_KEY / test_url
- `backend/api/admin/dashboard.py` — ZAI_API_KEY
- `backend/api/admin/usage.py` — glm-5.2 con precios reales
- `frontend/src/i18n/es.js` — `descAda: 'Análisis y rigor'`
- `frontend/src/i18n/en.js` — `descAda: 'Analysis & rigor'`
- `frontend/src/components/BottomBar/PipelineModal.jsx` — `capability: 'analysis'`

### Rollback disponible
```bash
# Timestamp: 20260627-044328
for f in api/chat.py api/admin/keys.py api/admin/dashboard.py api/admin/usage.py; do
  cp ~/jax-platform/backend/$f.backup-adaweb-20260627-044328 ~/jax-platform/backend/$f
done
```

### Incertidumbre declarada
El panel admin (botón *Probar*) usa `GET https://api.z.ai/api/paas/v4/models` con Bearer. No tengo confirmación de que Z.ai exponga ese endpoint con esa autenticación — si falla, reportará error de conexión pero no rompe el chat. Verificar manualmente en axioma-ia.io → Admin → API Keys → Z.ai → Probar.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
