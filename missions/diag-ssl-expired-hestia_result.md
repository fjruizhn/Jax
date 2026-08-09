# Resultado — diagnóstico SSL vencido: sol-lex.com y bdihn.com

Solo lectura. No se renovó, reinició ni modificó nada.

## Causa raíz (confirmada, no hipótesis)

**Ambos certificados son wildcard (`*.sol-lex.com`, `*.bdihn.com`), y ninguno
tiene `LETSENCRYPT: yes` en su configuración de Hestia** — a diferencia de
`monhagro.com` (referencia sana), que:
- NO es wildcard (cubre solo `monhagro.com` + `www.monhagro.com`)
- SÍ tiene `LETSENCRYPT: yes` explícito
- Fue emitido 2026-06-11, vence 2026-09-09 — ciclo de 90 días sano, recién
  renovado automáticamente.

Un certificado wildcard de Let's Encrypt **requiere desafío DNS-01**
(no HTTP-01). Hestia no soporta DNS-01 sin un plugin de proveedor DNS
configurado explícitamente. Sin `LETSENCRYPT: yes`, Hestia ni siquiera
considera estos dos dominios como gestionados por su renovador automático
— los trata como certificados "custom"/subidos manualmente una vez.

**Conclusión: estos certificados casi con certeza se emitieron una sola vez
por fuera del flujo normal de Hestia** (ej. `certbot` manual con DNS-01, o
subidos ya emitidos), y como Hestia nunca los marcó como
Let's-Encrypt-gestionados, jamás intentó renovarlos. No es que el
renovador esté "roto" — es que estos dos dominios nunca estuvieron
enganchados a él.

Confirmado también: no existe ningún cron de renovación SSL en
`/etc/cron.d/` (solo `hestia-proc` y `hestia-sftp`, ninguno de SSL) — el
mecanismo de auto-renovación de Hestia vive en otro lado (probablemente
crontab interno del usuario `admin`, no se pudo leer sin sudo interactivo),
pero es irrelevante para estos dos dominios específicamente ya que no
están marcados como gestionados.

## Factor agravante secundario (no la causa raíz, pero relevante para el fix)

Ambos dominios resuelven vía Cloudflare (IPs 104.21.x.x/172.67.x.x, igual
que `monhagro.com` — los 3 están proxiados). Al probar el path de ACME
challenge (`http://sol-lex.com/.well-known/acme-challenge/`) HOY:
- `sol-lex.com` → **HTTP 522** (Cloudflare: "connection timed out",
  no puede alcanzar el origen)
- `bdihn.com` → sin respuesta (timeout)

Esto significa que **incluso si se intentara una renovación HTTP-01 hoy
mismo** (ej. cambiando a un cert no-wildcard), probablemente fallaría
también por este problema de conectividad — que es independiente del tema
SSL y merece su propio diagnóstico (¿Cloudflare mal configurado apuntando
a una IP de origen vieja? ¿firewall del bridge bloqueando algo?
¿el propio bridge caído en ese aspecto?). `monhagro.com`, que sí funciona,
también está detrás de Cloudflare — así que no es "Cloudflare en general"
lo que falla, es algo específico del origen para estos 2 dominios.

## Recomendación (NO ejecutada — decisión de Fernando)

Dos caminos, no excluyentes:

1. **Más simple, probablemente lo correcto**: si ninguno de los dos
   dominios realmente necesita wildcard (ej. no usan subdominios reales
   además del apex), reemitir como certificados normales
   (`dominio.com` + `www.dominio.com`, como `monhagro.com`) vía el flujo
   normal de Hestia con `LETSENCRYPT: yes` — eso los engancha al
   auto-renovador y resuelve el problema de raíz para siempre, no solo
   una vez.
2. Si de verdad necesitan wildcard (hay subdominios reales en uso),
   hay que configurar DNS-01 en Hestia (requiere credenciales de API del
   proveedor DNS — a determinar cuál es, no investigado en esta misión).

Cualquiera de los dos caminos requiere primero resolver el 522/timeout de
Cloudflare→origen, o la renovación va a fallar igual.

## Explícitamente NO hecho

Ningún comando de renovación, ningún reinicio de servicio, ninguna
modificación de configuración en ningún dominio.
