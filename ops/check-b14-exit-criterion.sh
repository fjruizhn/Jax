#!/bin/bash
# Verifica el criterio de salida de B1.4 (Fase 1 credenciales): 7 dias
# consecutivos sin ninguna lectura source=env_fallback en los logs de
# jax-platform/jax-las-manos. Corre una sola vez, 2026-08-16, via systemd
# timer OnCalendar de fecha fija (ver check-b14-exit-criterion.timer).
# No modifica nada — solo lee logs y notifica por Telegram.
set -euo pipefail
set -a; source /etc/jax/.env; set +a

COUNT=$(journalctl -u jax-platform.service -u jax-las-manos.service \
  --since "7 days ago" --no-pager 2>/dev/null \
  | grep "credential_resolution" | grep -c "source=env_fallback" || true)

if [ "$COUNT" -eq 0 ]; then
  MSG="✅ B1.4 criterio de salida CUMPLIDO (0 lecturas env_fallback en 7 dias). PRs draft listos para revisar y decidir merge: https://github.com/fjruizhn/Jax/pull/1 · https://github.com/fjruizhn/Jax/pull/2 · https://github.com/fjruizhn/jax-platform/pull/1 · https://github.com/fjruizhn/jax-platform/pull/2"
else
  DETALLE=$(journalctl -u jax-platform.service -u jax-las-manos.service \
    --since "7 days ago" --no-pager 2>/dev/null \
    | grep "credential_resolution" | grep "source=env_fallback" \
    | grep -oP 'provider=\K[a-z_]+' | sort -u | tr '\n' ',' | sed 's/,$//')
  MSG="⚠️ B1.4 criterio de salida NO cumplido: $COUNT lecturas env_fallback en 7 dias (provider(s): ${DETALLE:-desconocido}). NO mergear los PRs de credenciales todavia — falta identificar que consumidor sigue cayendo al fallback de .env."
fi

echo "$MSG"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  -d "chat_id=${TELEGRAM_CHAT_ID}" \
  -d "text=${MSG}" > /dev/null

echo "$(date -Iseconds) $MSG" >> /home/fruiz/jax/ops/b14-exit-criterion.log
