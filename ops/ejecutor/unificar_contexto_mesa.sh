#!/usr/bin/env bash
# ops/ejecutor/unificar_contexto_mesa.sh — SP3: la Mesa y el Ejecutor en el MISMO contexto (131072).
#
# Por qué (medido, spec de Fase 2 §6.1/§6.2/§6.3): la Mesa usa el modelo a 32768 (el default de
# Ollama por VRAM) y el Ejecutor lo necesita a 131072; cada alternancia recarga (~3 s, 5-6 s en
# frío). Unificar le cuesta a la Mesa -2 % de lectura y +0,02/+0,26 s al primer token. El contexto
# sólo se cambia con un MODELO DERIVADO y un REBIND por el flujo aprobado: el PUT de bindings no
# acepta `params` y `/v1` no admite `num_ctx`. Nunca un UPDATE a mano.
#
# Consumidores de jax_local que quedan cubiertos por el rebind (resuelven el binding): el chat de
# la Mesa y `facet_canary` (jax-platform), LAS MANOS/Jacobs/REPL (jax facet_resolver).
#
# MODOS (ninguno reinicia servicios; ninguno toca .10/.11/.20):
#   --probar-respaldo      dump de facet_binding y model + restauración en un MariaDB DESCARTABLE
#                          (docker, sin red) + comparación de CHECKSUM. Sólo LEE producción.
#   --aplicar              precondiciones → uso cero → respaldo probado → derivado → sync de
#                          catálogo → contrato de dispatch → rebind → verificación. Si la
#                          verificación falla, REVIERTE sola al modelo base.
#   --revertir DIR         rebind de la faceta al modelo base guardado en DIR y verificación.
#   --restaurar-dump DIR   ÚLTIMO RECURSO: reimporta facet_binding desde el dump de DIR. Exige
#                          JAX_SP3_CONFIRMO_RESTAURAR_DUMP=<nombre de DIR> y estampa el sello.
#
# ENTORNO: /etc/jax/.env (DB, JAX_OLLAMA_URL, JAX_PLATFORM_URL) + ops/ejecutor/sp3_entorno.conf.
#   set -a; . <(sudo -n cat /etc/jax/.env); . ops/ejecutor/sp3_entorno.conf; set +a
# Para --aplicar/--revertir: JAX_ADMIN_TOKEN_ARCHIVO = archivo 0600 con el access token de un
# superadmin (dura 15 min). El token nunca va en argv (`ps` lo vería): va en una cabecera leída
# de un archivo temporal 0600.
set -euo pipefail
umask 077

MODO="${1:-}"
: "${JAX_DB_USER:?}" "${JAX_DB_PASSWORD:?}" "${JAX_DB_NAME:?}" "${JAX_DB_HOST:?}" "${JAX_DB_PORT:?}"
: "${JAX_MESA_FACETA:?}"
RESPALDOS="${JAX_SP3_RESPALDOS:-$HOME/backups}"
MARCA="$(date +%Y%m%d-%H%M%S)"
TMP="$(mktemp -d)"
CONTENEDOR=""
limpiar() {
  if [ -n "$CONTENEDOR" ]; then sudo -n docker rm -f "$CONTENEDOR" >/dev/null 2>&1 || true; fi
  rm -rf "$TMP"
}
trap limpiar EXIT

log() { echo "$*"; }
morir() { echo "$*" >&2; exit 1; }

[[ "$JAX_MESA_FACETA" =~ ^[a-z_]+$ ]] || morir "faceta_invalida"
sql() { MYSQL_PWD="$JAX_DB_PASSWORD" mariadb -h"$JAX_DB_HOST" -P"$JAX_DB_PORT" -u"$JAX_DB_USER" "$JAX_DB_NAME" -N -B -e "$1"; }
checksum_tablas() {  # "tabla<TAB>checksum" sin el prefijo de la base
  sql "CHECKSUM TABLE facet_binding, model EXTENDED" | sed -E 's/^[^.]+\.//'
}

# ---------------------------------------------------------------------------------------------
# Respaldo y restauración probada
# ---------------------------------------------------------------------------------------------
respaldar() {
  DIR="$RESPALDOS/sp3-unificar-$MARCA"
  mkdir -p "$DIR"
  checksum_tablas > "$DIR/checksum_antes.tsv"
  for t in facet_binding model; do
    MYSQL_PWD="$JAX_DB_PASSWORD" mariadb-dump -h"$JAX_DB_HOST" -P"$JAX_DB_PORT" -u"$JAX_DB_USER" \
      --single-transaction --no-tablespaces "$JAX_DB_NAME" "$t" > "$DIR/dump_$t.sql"
  done
  checksum_tablas > "$DIR/checksum_despues.tsv"
  # Si alguien escribió en medio, el dump no corresponde a ningún checksum: no vale.
  cmp -s "$DIR/checksum_antes.tsv" "$DIR/checksum_despues.tsv" || morir "tablas_cambiaron_durante_el_dump dir=$DIR"
  sql "SELECT b.model_ref, b.provider_id, b.model_id, m.max_tokens_param, m.max_output_tokens
       FROM facet_binding b JOIN model m ON m.id = b.model_ref
       WHERE b.facet_key='$JAX_MESA_FACETA' AND b.role='primary'" > "$DIR/binding_antes.tsv"
  [ "$(wc -l < "$DIR/binding_antes.tsv")" = 1 ] || morir "binding_primario_no_unico dir=$DIR"
  curl -sS --max-time 10 "${JAX_OLLAMA_URL%/}/api/ps" > "$DIR/ollama_ps_antes.json" || true
  (cd "$DIR" && sha256sum dump_*.sql checksum_antes.tsv binding_antes.tsv > SHA256SUMS)
  log "respaldo=$DIR"
}

probar_restauracion() {
  local version imagen clave i
  version="$(sql "SELECT VERSION()" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+')"
  imagen="mariadb:$version"
  clave="$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9')"
  CONTENEDOR="sp3-restauracion-$MARCA"
  sudo -n docker run -d --rm --network none --name "$CONTENEDOR" \
    -e MARIADB_ROOT_PASSWORD="$clave" -e MARIADB_DATABASE=prueba "$imagen" >/dev/null
  for i in $(seq 1 60); do
    if sudo -n docker exec -e MYSQL_PWD="$clave" "$CONTENEDOR" mariadb -uroot -e "SELECT 1" prueba >/dev/null 2>&1; then break; fi
    sleep 1
  done
  for t in facet_binding model; do
    sudo -n docker exec -i -e MYSQL_PWD="$clave" "$CONTENEDOR" mariadb -uroot prueba < "$DIR/dump_$t.sql"
  done
  sudo -n docker exec -e MYSQL_PWD="$clave" "$CONTENEDOR" mariadb -uroot -N -B prueba \
    -e "CHECKSUM TABLE facet_binding, model EXTENDED" | sed -E 's/^[^.]+\.//' > "$DIR/checksum_restaurado.tsv"
  sudo -n docker rm -f "$CONTENEDOR" >/dev/null
  CONTENEDOR=""
  if ! cmp -s "$DIR/checksum_antes.tsv" "$DIR/checksum_restaurado.tsv"; then
    diff "$DIR/checksum_antes.tsv" "$DIR/checksum_restaurado.tsv" >&2 || true
    morir "restauracion_distinta dir=$DIR"
  fi
  (cd "$DIR" && sha256sum checksum_restaurado.tsv >> SHA256SUMS)
  log "restauracion_probada=true imagen=$imagen $(tr '\t\n' '= ' < "$DIR/checksum_restaurado.tsv")"
}

# ---------------------------------------------------------------------------------------------
# API de jax-platform (superadmin) y Ollama
# ---------------------------------------------------------------------------------------------
preparar_token() {
  : "${JAX_ADMIN_TOKEN_ARCHIVO:?}" "${JAX_PLATFORM_URL:?}"
  [ "$(stat -c '%a' "$JAX_ADMIN_TOKEN_ARCHIVO")" = 600 ] || morir "token_archivo_no_0600"
  printf 'Authorization: Bearer %s\n' "$(tr -d '[:space:]' < "$JAX_ADMIN_TOKEN_ARCHIVO")" > "$TMP/cabecera"
}

api() {  # api METODO RUTA [CUERPO] -> imprime el código HTTP; el cuerpo queda en $TMP/respuesta.json
  local args=(-sS --max-time 120 -o "$TMP/respuesta.json" -w '%{http_code}' -X "$1" -H @"$TMP/cabecera")
  if [ -n "${3:-}" ]; then args+=(-H 'content-type: application/json' --data "$3"); fi
  curl "${args[@]}" "${JAX_PLATFORM_URL%/}$2"
}

contextos_cargados() {  # contextos_cargados MODELO -> los context_length con que Ollama lo tiene cargado
  curl -sS --max-time 10 "${JAX_OLLAMA_URL%/}/api/ps" | python3 -c '
import json, sys
nombre = sys.argv[1]
for m in json.load(sys.stdin).get("models", []):
    if m.get("name") in (nombre, nombre + ":latest"):
        print(m.get("context_length"))
' "$1"
}

esperar_sonda_ok() {  # esperar_sonda_ok DESDE_EPOCH TOPE_S
  local fin=$(( $(date +%s) + $2 )) resultado
  while [ "$(date +%s)" -lt "$fin" ]; do
    resultado="$(sql "SELECT outcome FROM facet_health_event WHERE facet='$JAX_MESA_FACETA'
                      AND source='canary_rebind' AND ts > $1 ORDER BY ts DESC LIMIT 1")"
    if [ "$resultado" = ok ]; then return 0; fi
    if [ -n "$resultado" ]; then log "sonda_rebind=$resultado"; return 1; fi
    sleep 5
  done
  log "sonda_rebind=sin_resultado"
  return 1
}

rebind() {  # rebind MODEL_REF -> exige 200 y lo verifica en la tabla
  local codigo
  codigo="$(api PUT "/api/admin/facet-bindings/$JAX_MESA_FACETA" \
            "{\"provider_id\":\"ollama\",\"model_ref\":$1,\"role\":\"primary\"}")"
  [ "$codigo" = 200 ] || { cat "$TMP/respuesta.json" >&2; return 1; }
  [ "$(sql "SELECT model_ref FROM facet_binding WHERE facet_key='$JAX_MESA_FACETA' AND role='primary'")" = "$1" ]
}

# ---------------------------------------------------------------------------------------------
# Revertir: al modelo base del respaldo
# ---------------------------------------------------------------------------------------------
revertir() {
  local dir="$1" base_ref base_modelo desde
  (cd "$dir" && sha256sum -c --quiet --ignore-missing SHA256SUMS) || morir "respaldo_alterado dir=$dir"
  IFS=$'\t' read -r base_ref _ base_modelo _ _ < "$dir/binding_antes.tsv"
  desde="$(date +%s)"
  rebind "$base_ref" || morir "revertir_rebind_fallo base_ref=$base_ref"
  log "revertido_binding=$base_modelo model_ref=$base_ref"
  if esperar_sonda_ok "$desde" 300; then log "revertido_sonda=ok"; else log "revertido_sonda=no_ok"; fi
  if [ "${JAX_PROXY_CARRIL_MODELO:-}" != "$base_modelo" ]; then
    # El proxy seguiría fijado al derivado: la jaula no podría trabajar, pero tampoco desalojaría a la Mesa.
    log "aviso=proxy_fijado_a_otro_modelo proxy=${JAX_PROXY_CARRIL_MODELO:-} binding=$base_modelo"
  fi
}

# ---------------------------------------------------------------------------------------------
# Aplicar
# ---------------------------------------------------------------------------------------------
aplicar() {
  : "${JAX_OLLAMA_URL:?}" "${JAX_MESA_MODELO_DERIVADO:?}" "${JAX_MESA_NUM_CTX:?}" "${JAX_PROXY_CARRIL_MODELO:?}"
  local derivado="$JAX_MESA_MODELO_DERIVADO" ctx="$JAX_MESA_NUM_CTX"
  local base_ref proveedor base_modelo param tope nuevo_ref codigo desde usos contrato
  [[ "$derivado" =~ ^[A-Za-z0-9._-]+$ ]] || morir "derivado_invalido"
  [[ "$ctx" =~ ^[0-9]+$ ]] || morir "num_ctx_invalido"
  # El proxy tiene que fijar EL MISMO modelo que va a usar la Mesa, o la jaula recarga en cada petición.
  [ "$JAX_PROXY_CARRIL_MODELO" = "$derivado" ] || morir "proxy_y_mesa_con_modelos_distintos"
  preparar_token
  [ "$(api GET /api/admin/facet-bindings)" = 200 ] || morir "token_sin_superadmin"

  # Cortes de jax_local sólo con cero uso real (LEDGER): el rebind recarga el modelo (~3-6 s).
  usos="$(sql "SELECT COUNT(*) FROM axioma_usage WHERE facet='$JAX_MESA_FACETA' AND created_at > NOW() - INTERVAL 30 MINUTE")"
  [ "$usos" = 0 ] || morir "uso_real_reciente=$usos"

  respaldar
  probar_restauracion
  IFS=$'\t' read -r base_ref proveedor base_modelo param tope < "$DIR/binding_antes.tsv"
  [ "$proveedor" = ollama ] || morir "binding_no_es_ollama=$proveedor"
  if [ "$base_modelo" = "$derivado" ]; then log "ya_unificado=true"; return 0; fi

  # 1. Derivado: mismos pesos, contexto fijo. No carga el modelo.
  printf 'FROM %s\nPARAMETER num_ctx %s\n' "$base_modelo" "$ctx" > "$DIR/Modelfile"
  ollama create "$derivado" -f "$DIR/Modelfile"
  ollama show "$derivado" --parameters | grep -qE "^num_ctx[[:space:]]+$ctx$" || morir "derivado_sin_num_ctx"
  log "derivado=$derivado desde=$base_modelo num_ctx=$ctx"

  # 2. Catálogo: el sync de Ollama escribe la fila (sólo `model`, nunca bindings).
  codigo="$(api POST /api/admin/models/sync)"
  [ "$codigo" = 200 ] || { cat "$TMP/respuesta.json" >&2; morir "sync_fallo=$codigo"; }
  #    El sync guarda el nombre como lo lista Ollama, CON tag (`<derivado>:latest`; visto en la
  #    corrida real del 2026-09-17). Se aceptan las dos formas, pero UNA sola fila: dos filas del
  #    mismo modelo son un catálogo ambiguo y no se elige a ciegas.
  nuevo_ref="$(sql "SELECT id FROM model WHERE provider_id='ollama' AND model_id IN ('$derivado', '$derivado:latest') AND status='available'")"
  [[ "$nuevo_ref" =~ ^[0-9]+$ ]] || morir "derivado_no_esta_en_el_catalogo filas=$(printf '%s' "$nuevo_ref" | grep -c . || true)"

  # 3. Contrato de dispatch: el mismo que el modelo base (leído del respaldo, no escrito acá).
  #    Transporte `ollama`: el base no declara contrato (NULL, NULL) y el PUT rechaza uno vacío
  #    con 422 (visto en la corrida real del 2026-09-17). Sin contrato en el base, no se copia:
  #    el derivado queda igual que el base, que es lo que se busca.
  if [ "$param" = NULL ] && [ "$tope" = NULL ]; then
    log "catalogo model_ref=$nuevo_ref contrato=sin_contrato_en_el_base"
  else
    contrato="$(python3 -c '
import json, sys
p, t = sys.argv[1], sys.argv[2]
print(json.dumps({"max_tokens_param": None if p == "NULL" else p, "max_output_tokens": None if t == "NULL" else int(t)}))
' "$param" "$tope")"
    codigo="$(api PUT "/api/admin/models/$nuevo_ref/contrato-dispatch" "$contrato")"
    [ "$codigo" = 200 ] || { cat "$TMP/respuesta.json" >&2; morir "contrato_fallo=$codigo"; }
    log "catalogo model_ref=$nuevo_ref contrato=$contrato"
  fi

  # 4. Rebind por el endpoint aprobado. Desde acá, cualquier fallo revierte al base.
  desde="$(date +%s)"
  if ! rebind "$nuevo_ref"; then
    log "fallo=rebind revirtiendo"; revertir "$DIR"; morir "unificado=false"
  fi
  log "rebind=$JAX_MESA_FACETA model_ref=$nuevo_ref"

  # 5. Verificación: la sonda por rebind (encolada por el PUT) respondió ok con el derivado, y
  #    Ollama tiene el derivado a $ctx y NO el base a otro contexto (eso sería una recarga pendiente).
  if ! esperar_sonda_ok "$desde" 300; then
    log "fallo=sonda revirtiendo"; revertir "$DIR"; morir "unificado=false"
  fi
  if [ "$(contextos_cargados "$derivado")" != "$ctx" ] || [ -n "$(contextos_cargados "$base_modelo" | grep -vx "$ctx" || true)" ]; then
    log "fallo=ollama_ps derivado=$(contextos_cargados "$derivado" | tr '\n' ,) base=$(contextos_cargados "$base_modelo" | tr '\n' ,) revirtiendo"
    revertir "$DIR"; morir "unificado=false"
  fi
  curl -sS --max-time 10 "${JAX_OLLAMA_URL%/}/api/ps" > "$DIR/ollama_ps_despues.json"
  echo "$nuevo_ref" > "$DIR/VERIFICADO"
  log "unificado=true faceta=$JAX_MESA_FACETA modelo=$derivado num_ctx=$ctx respaldo=$DIR"
  log "rollback: $0 --revertir $DIR"
}

restaurar_dump() {
  local dir="$1"
  : "${JAX_SP3_CONFIRMO_RESTAURAR_DUMP:?}"
  [ "$JAX_SP3_CONFIRMO_RESTAURAR_DUMP" = "$(basename "$dir")" ] || morir "confirmacion_no_coincide"
  (cd "$dir" && sha256sum -c --quiet --ignore-missing SHA256SUMS) || morir "respaldo_alterado dir=$dir"
  MYSQL_PWD="$JAX_DB_PASSWORD" mariadb -h"$JAX_DB_HOST" -P"$JAX_DB_PORT" -u"$JAX_DB_USER" "$JAX_DB_NAME" \
    < "$dir/dump_facet_binding.sql"
  # El sello invalida la resolución cacheada de facetas en la Mesa, LAS MANOS y el REPL.
  (cd "${JAX_PLATFORM_REPO_ROOT:-$HOME/jax-platform}/backend" && .venv/bin/python -c 'import facet_resolver; facet_resolver._tocar_sello()')
  if diff <(sql "CHECKSUM TABLE facet_binding EXTENDED" | sed -E 's/^[^.]+\.//') \
          <(grep '^facet_binding' "$dir/checksum_antes.tsv"); then
    log "facet_binding_restaurada=true"
  else
    morir "facet_binding_restaurada=false"
  fi
}

case "$MODO" in
  --probar-respaldo) respaldar; probar_restauracion ;;
  --aplicar) aplicar ;;
  --revertir) [ -n "${2:-}" ] || morir "falta_dir"; preparar_token; revertir "$2" ;;
  --restaurar-dump) [ -n "${2:-}" ] || morir "falta_dir"; restaurar_dump "$2" ;;
  *) echo "uso: $0 --probar-respaldo | --aplicar | --revertir DIR | --restaurar-dump DIR" >&2; exit 2 ;;
esac
