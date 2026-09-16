#!/usr/bin/env bash
# Fase 0 · U6: ¿el bucket de respaldos tiene candado de verdad?
# Sube un objeto canario diminuto y trata de borrarlo (gate G4).
# Las credenciales van a curl por stdin (-K -), nunca por argv.
set -euo pipefail
ENVF="${R2_ENV:-/etc/restic/r2.env}"
set -a; source "$ENVF"; set +a
url="${RESTIC_REPOSITORY#s3:}"
endpoint="$(printf '%s' "$url" | cut -d/ -f1-3)"
bucket="$(printf '%s' "$url" | cut -d/ -f4)"
clave="fase0-canario/canario-$(date +%Y%m%dT%H%M%S).txt"
cfg() { printf 'user = "%s:%s"\n' "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY"; }
S3=(--aws-sigv4 "aws:amz:auto:s3" -K -)
cuerpo="$(mktemp)"
echo "bucket=$bucket clave=$clave"
echo -n "PUT    -> "; cfg | curl -sS -o /dev/null -w '%{http_code}\n' "${S3[@]}" -X PUT --data-binary "canario fase0 ejecutor $(date -Is)" "$endpoint/$bucket/$clave"
echo -n "DELETE -> "; cfg | curl -sS -o "$cuerpo" -w '%{http_code}\n' "${S3[@]}" -X DELETE "$endpoint/$bucket/$clave"
echo "cuerpo del DELETE:"; cat "$cuerpo"; echo
echo -n "HEAD   -> "; cfg | curl -sS -o /dev/null -I -w '%{http_code}\n' "${S3[@]}" "$endpoint/$bucket/$clave"
rm -f "$cuerpo"
