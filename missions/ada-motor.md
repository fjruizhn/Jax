# Misión Hyde — Conectar a Ada (GLM-5.2 / Z.ai) como séptima faceta

> Ejecutar con: `jax --task ~/jax/missions/ada-motor.md`
> Hipatia inicia con `/using-superpowers` y `/ruflo`.
> Plugins: superpowers, ui-ux-pro-max, ruflo-core, impeccable, token-optimizer, napkin.

---

## 1. Objetivo

Registrar a **Ada** como séptima faceta de JAX/Axioma, conectándola al modelo
**GLM-5.2** vía la API directa de Z.ai (OpenAI-compatible). Ada es faceta
conversacional/analítica, no herramienta de coding-plan.

## 2. Principios (PROTOCOLO HYDE activo)

- **No suponer** — "El que supone se equivoca." No asumir el schema del Motor
  Registry: leerlo primero.
- **Ningún comando sin output conocido.** Cada paso declara su salida esperada
  antes de ejecutarse.
- **Backup antes de modificar** cualquier archivo de config, .env o Registry.
- **No hardcoding.** La key vive en `.env`, nunca en código ni en el Registry.
- **Toda incertidumbre se declara explícita** en el reporte final. Ningún
  diagnóstico se cierra con "probablemente".
- **Causa raíz, no parches.** Si algo falla, se resuelve de fondo.

## 3. Datos verificados (externos — NO re-investigar, ya confirmados)

| Campo        | Valor |
|--------------|-------|
| Proveedor    | Z.ai (plataforma overseas, **no** bigmodel.cn) |
| Protocolo    | OpenAI-compatible (chat completions, function calling, streaming) |
| `base_url`   | `https://api.z.ai/api/paas/v4/` |
| `model`      | `glm-5.2` |
| Auth         | `Authorization: Bearer $ZAI_API_KEY` |
| Billing      | Saldo prepago / pay-as-you-go (NO Coding Plan) |

**TRAMPA CONOCIDA:** existe un endpoint aparte `https://api.z.ai/api/coding/paas/v4`
que es **exclusivo** para cuota de Coding Plan y herramientas tipo Claude Code.
Ada **NO** lo usa. Si Hipatia ve ese path en cualquier config → es error, corregir.

**Extensiones opcionales de Z.ai** (no estándar OpenAI): parámetro `thinking`
(`{"type": "enabled" | "disabled"}`) y `reasoning_effort`. NO incluirlas en la
primera prueba; integrarlas solo después de confirmar el camino básico.

## 4. Tareas (orden estricto)

### 4.1 — Reconocimiento (read-only)
- Localizar y **leer** un motor OpenAI-compatible ya existente como referencia
  (ej. el de DeepSeek o el de Thot). Documentar:
  - estructura/schema del motor,
  - cómo se registra una faceta en el Motor Registry,
  - convención de nombres de las variables de `.env`,
  - dónde se define el system prompt / persona de cada faceta.
- **No tocar nada todavía.** Salida esperada: un resumen del patrón a replicar.

### 4.2 — Smoke test aislado de la key (antes de tocar JAX)
Validar la credencial con una llamada mínima, fuera del orquestador:

```bash
curl -sS -X POST "https://api.z.ai/api/paas/v4/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ZAI_API_KEY" \
  -d '{
    "model": "glm-5.2",
    "messages": [
      {"role": "user", "content": "Responde solo: OK Ada en línea."}
    ]
  }'
```

- **Output esperado:** HTTP 200 + `choices[0].message.content` con texto.
- **Si 401/403:** parar. La key o el billing están mal (¿Coding Plan en vez de
  saldo general?). Reportar, no continuar.

### 4.3 — `.env`
- Hacer **backup** del `.env`.
- Agregar `ZAI_API_KEY=<valor>` siguiendo la convención de nombres existente
  (ajustar el nombre si los otros motores usan otro patrón, p.ej. `*_API_KEY`).
- Verificar que `.env` esté en `.gitignore`.

### 4.4 — Crear el motor de Ada
- Clonar el patrón del motor de referencia (4.1), cambiando **solo**:
  `base_url` → `https://api.z.ai/api/paas/v4/`, `model` → `glm-5.2`,
  key → `ZAI_API_KEY`.
- No reinventar el adapter: si el motor de referencia ya es OpenAI-compatible,
  Ada reusa el mismo cliente.

### 4.5 — Registrar a Ada en el Motor Registry (faceta #7)
- Backup del Registry antes de modificar.
- Alta de la faceta `ada` con la persona de la sección 5.
- Si hay etiquetas visibles en frontend, respetar **i18n (ES/EN)** y no hardcodear.

### 4.6 — Test de routing end-to-end
- Enviar un prompt conocido a Ada vía `/api/chat` y confirmar:
  1. el router selecciona a Ada cuando se la invoca,
  2. responde con contenido de `glm-5.2`,
  3. la memoria semántica registra el intercambio en el scope correcto.
- **Output esperado:** documentar la respuesta y el facet seleccionado.

### 4.7 — Reporte final
- Archivos tocados + diffs.
- Resultado de 4.2 y 4.6 (con la respuesta real, no parafraseada).
- Incertidumbres declaradas explícitamente.
- Rollback disponible (ver sección 6).

## 5. Persona de Ada (system prompt)

> *Ratificar o redlinear antes de ejecutar.*

```
Eres Ada, séptima faceta del concilio JAX/Axioma, nombrada por Ada Lovelace,
autora del primer algoritmo de la historia.

Tu dominio es el rigor analítico y la formalización. Tomas la intención y la
conviertes en algoritmo preciso; ves la estructura debajo del ruido; distingues
lo demostrado de lo asumido. Practicas la "poetical science" de Lovelace:
imaginación disciplinada por la prueba.

Principios operativos:
- "El que supone se equivoca." Nunca cierres un razonamiento sobre una
  suposición no declarada. Si falta un dato, lo nombras; no lo rellenas.
- Exiges la derivación, no la afirmación. Cuando algo es incierto, lo marcas
  como incierto.
- Eres precisa y elegante, no verbosa. Vas a la estructura del problema.
- Complementas al concilio: donde otros narran o exploran, tú formalizas y
  verificas.

Respondes en el idioma del usuario (ES por defecto). Mantienes la voz serena y
exacta de quien confía en la lógica más que en la retórica.
```

## 6. Rollback

- Restaurar `.env`, el Motor Registry y cualquier config desde los backups de
  esta misión.
- Quitar la entrada de la faceta `ada` del Registry.
- La cuenta/key de Z.ai queda intacta (es externa).

## 7. Criterios de aceptación

- [ ] Smoke test (4.2) devuelve 200 con contenido.
- [ ] `ZAI_API_KEY` en `.env`, fuera de git, sin hardcoding.
- [ ] Motor Ada creado siguiendo el patrón existente.
- [ ] Ada registrada como faceta #7.
- [ ] Test de routing (4.6) verde con respuesta real documentada.
- [ ] Reporte final con incertidumbres declaradas.
