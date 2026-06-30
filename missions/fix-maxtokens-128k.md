# Misión Hyde — Subir `max_tokens` al techo real de GLM-5.2 (131072)

> Ejecutar con: `jax --task ~/jax/missions/fix-maxtokens-128k.md --facet hyde`
> Hipatia inicia con `/using-superpowers` y `/ruflo`.

---

## 1. Objetivo y causa raíz (verificada empíricamente)

`max_tokens` en el worker OpenAI-compat está en 32000. **No es un límite de corte:
es la RESERVA de espacio de salida** (el modelo para solo con `finish_reason: stop`).
El problema era que 32000 se COMPARTE entre el razonamiento interno de GLM-5.2
(`reasoning_tokens`) y el texto visible (`content`), así que en documentos largos el
razonamiento consumía presupuesto y el texto se truncaba.

**Verificado por curl crudo contra el endpoint de Z.ai:**
- Z.ai acepta `max_tokens` hasta **131072** (128K) para glm-5.2 (probado 32K/64K/98K/131K, todos OK).
- Con `max_tokens: 131072` y razonamiento ENCENDIDO, GLM generó una lista 1→3000
  (10.347 tokens, 937 de razonamiento) con `finish_reason: stop` — completa, sin corte.

**Fix:** subir `max_tokens` de 32000 a **131072** en el worker OpenAI-compat (Ada + Kimi).
Esto le da la "pista completa": razona a fondo Y escribe documentos largos completos,
sin pelear por presupuesto. NO se toca el razonamiento (es el valor del modelo, se deja libre).

`max_tokens` alto es seguro: solo reserva; se paga/tarda por lo realmente generado.
El `task_timeout` ya está en 3600s, suficiente para generaciones largas.

## 2. Principios (HYDE activo)
- No suponer; leer antes de editar. Backup antes de modificar.
- Ningún cambio sin prueba (gate §5). Declarar incertidumbres.

## 3. Reconocimiento (read-only) — NO EDITAR TODAVÍA
```bash
# Dónde está el max_tokens actual (lo pusimos en una misión previa)
grep -n "max_tokens" ~/jax/jax/muscles/base.py
```
**Reportá** las líneas exactas con `"max_tokens": 32000` (deberían ser 2: una en el
worker DeepSeek ~212 y otra en el OpenAI-compat ~246).

## 4. Cambio — `~/jax/jax/muscles/base.py`
- Backup `base.py.backup-128k-$TS`.
- Cambiar `"max_tokens": 32000` → `"max_tokens": 131072` en el worker **OpenAI-compat**
  (`_call_openai`, el que usan Ada y Kimi).
- Aplicar el MISMO cambio al worker **DeepSeek** (`_call_deepseek`, Jekyll) por
  consistencia — mismo techo latente, mismo beneficio. (Si su endpoint rechazara
  131072, declararlo y dejar DeepSeek en un valor que acepte; reportar.)
- NO tocar nada más del payload (streaming, modelo, etc. quedan como están).

## 5. Gate de prueba (obligatorio)
```bash
# 1) Salida LARGA por el camino normal de JAX (Ada), razonamiento encendido:
cat > ~/maxtok128-gate.md <<'EOF'
ada: generá una lista numerada del 1 al 1500, un número por línea, sin texto extra.
En la última línea escribí exactamente: GATE-128K-OK
EOF
jax --task ~/maxtok128-gate.md --facet ada
tail -4 ~/maxtok128-gate_result.md
wc -l ~/maxtok128-gate_result.md
#   ESPERADO: la lista llega a 1500 + GATE-128K-OK al final, sin corte.

# 2) No-regresión salida corta:
cat > ~/maxtok128-short.md <<'EOF'
ada: respondé solo SHORT-OK
EOF
jax --task ~/maxtok128-short.md --facet ada
cat ~/maxtok128-short_result.md
#   ESPERADO: SHORT-OK.

# 3) Verificación en disco:
grep -n "max_tokens" ~/jax/jax/muscles/base.py
#   ESPERADO: 131072 en el/los worker(s).
```
**Criterio de aceptación:** (1) lista a 1500 + GATE-128K-OK sin corte; (2) SHORT-OK;
(3) `131072` presente en el código. Si la lista se corta antes de 1500 → reportar
`finish_reason` real y NO dar por bueno.

## 6. Reporte final
Archivo tocado + diff (las líneas de max_tokens), resultado del gate (¿llegó a 1500?),
si DeepSeek aceptó 131072 o no, incertidumbres, rollback (`base.py.backup-128k-$TS`).
