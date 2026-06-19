#!/bin/bash
# Motor Registry v0.2 test suite — LAS MANOS
# Runs from /home/fruiz/jax/workspace

set -u
BASE_URL="http://127.0.0.1:7777"
PAUSE_PATH="/etc/jax/PAUSE"

DISPATCH_PAYLOAD='{"caller": "hyde", "capability": "refactor", "prompt": "refactoriza el archivo de prueba: def suma(a, b): return a + b"}'

PASS="[PASS]"
FAIL="[FAIL]"

echo "============================================================"
echo "HEALTH CHECK"
echo "============================================================"
HEALTH=$(curl -s --max-time 5 "$BASE_URL/health")
echo "Response: $HEALTH"
if echo "$HEALTH" | grep -q '"alive"'; then
    echo "$PASS — servidor vivo"
else
    echo "$FAIL — /health no respondió correctamente"
    exit 1
fi

echo ""
echo "============================================================"
echo "KIMI_API_KEY — verificación en proceso del servidor"
echo "============================================================"
# Comprobar via /proc si KIMI_API_KEY está en el entorno del proceso uvicorn
UVICORN_PID=$(pgrep -f "uvicorn server:app" | head -1)
if [ -n "$UVICORN_PID" ]; then
    echo "PID uvicorn: $UVICORN_PID"
    if sudo -n strings /proc/$UVICORN_PID/environ 2>/dev/null | grep -q "KIMI_API_KEY"; then
        echo "$PASS — KIMI_API_KEY presente en entorno del proceso"
    else
        echo "  KIMI_API_KEY no encontrada en /proc/$UVICORN_PID/environ (puede requerir sudo)"
    fi
fi

echo ""
echo "============================================================"
echo "PRUEBA 1 — POST /motor/dispatch"
echo "============================================================"
RESP1=$(curl -s --max-time 15 -X POST "$BASE_URL/motor/dispatch" \
    -H 'Content-Type: application/json' \
    -d "$DISPATCH_PAYLOAD")
echo "Response: $RESP1"

JOB_ID=$(echo "$RESP1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null)
DISPATCH_STATUS=$(echo "$RESP1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
echo "job_id: $JOB_ID"
echo "status: $DISPATCH_STATUS"

if [ -n "$JOB_ID" ] && [ "$DISPATCH_STATUS" = "pending" -o "$DISPATCH_STATUS" = "running" ]; then
    echo "$PASS — job_id obtenido, status=$DISPATCH_STATUS"
elif [ "$DISPATCH_STATUS" = "rejected" ]; then
    REASON=$(echo "$RESP1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('rejected_reason','?'))" 2>/dev/null)
    echo "$FAIL — dispatch rechazado: $REASON"
    exit 1
else
    echo "$FAIL — status inesperado: $DISPATCH_STATUS"
    exit 1
fi

echo ""
echo "============================================================"
echo "PRUEBA 2 — Polling GET /motor/job/$JOB_ID"
echo "============================================================"
MAX_POLLS=24
FINAL_VIEW=""
FINAL_STATUS=""
START_TIME=$(date +%s)

for i in $(seq 1 $MAX_POLLS); do
    VIEW=$(curl -s --max-time 10 "$BASE_URL/motor/job/$JOB_ID")
    STATUS=$(echo "$VIEW" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null)
    ELAPSED=$(( $(date +%s) - START_TIME ))
    echo "  Poll $i (${ELAPSED}s): status=$STATUS"

    if [ "$STATUS" != "running" ] && [ "$STATUS" != "pending" ]; then
        FINAL_VIEW="$VIEW"
        FINAL_STATUS="$STATUS"
        break
    fi
    sleep 5
done

if [ -z "$FINAL_VIEW" ]; then
    echo "$FAIL — timeout (120s) sin resultado"
else
    echo ""
    echo "Respuesta final:"
    echo "$FINAL_VIEW" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2, ensure_ascii=False))" 2>/dev/null || echo "$FINAL_VIEW"
    echo ""

    # Check status = completed
    if [ "$FINAL_STATUS" = "completed" ]; then
        echo "$PASS — status=completed"
    else
        echo "$FAIL — status=$FINAL_STATUS (esperado: completed)"
    fi

    # Check result_summary is real text, not stub
    RESULT_SUMMARY=$(echo "$FINAL_VIEW" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result_summary','') or '')" 2>/dev/null)
    if [ -n "$RESULT_SUMMARY" ] && ! echo "$RESULT_SUMMARY" | grep -q "\[STUB"; then
        echo "$PASS — result_summary contiene texto real de Kimi (${#RESULT_SUMMARY} chars)"
        echo "  Preview: ${RESULT_SUMMARY:0:100}..."
    elif echo "$RESULT_SUMMARY" | grep -q "\[STUB"; then
        echo "$FAIL — result_summary contiene stub: $RESULT_SUMMARY"
    else
        echo "$FAIL — result_summary vacío o ausente"
        # Print the full error if any
        ERROR=$(echo "$FINAL_VIEW" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',''))" 2>/dev/null)
        [ -n "$ERROR" ] && echo "  error: $ERROR"
    fi

    # Check reasoning_content NOT in response
    if echo "$FINAL_VIEW" | grep -q "reasoning_content"; then
        echo "$FAIL — reasoning_content APARECE en la respuesta JSON (fuga de datos)"
    else
        echo "$PASS — reasoning_content NO aparece en la respuesta JSON"
    fi
fi

echo ""
echo "============================================================"
echo "PRUEBA 3 — Kill switch (/etc/jax/PAUSE)"
echo "============================================================"

# 3a — asegurar que PAUSE no existe
sudo rm -f "$PAUSE_PATH"
echo "3a. sudo rm -f $PAUSE_PATH — OK"

# 3b — crear PAUSE
sudo mkdir -p /etc/jax
sudo touch "$PAUSE_PATH"
if [ -f "$PAUSE_PATH" ]; then
    echo "3b. PAUSE creado: $PASS"
else
    echo "3b. PAUSE NO creado: $FAIL"
fi

# Verificar en /health que kill_switch_active=true
HEALTH_KS=$(curl -s --max-time 5 "$BASE_URL/health")
KS_ACTIVE=$(echo "$HEALTH_KS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('kill_switch_active','?'))" 2>/dev/null)
echo "   /health kill_switch_active=$KS_ACTIVE"

# 3c — POST dispatch
echo ""
echo "3c. POST /motor/dispatch con PAUSE activo..."
RESP3=$(curl -s --max-time 15 -X POST "$BASE_URL/motor/dispatch" \
    -H 'Content-Type: application/json' \
    -d "$DISPATCH_PAYLOAD")
echo "    Dispatch response: $RESP3"
JOB_ID3=$(echo "$RESP3" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null)
echo "    job_id3: $JOB_ID3"

# 3d — GET job status
echo ""
echo "3d. Esperando 3s y GET /motor/job/$JOB_ID3..."
sleep 3
VIEW3=$(curl -s --max-time 10 "$BASE_URL/motor/job/$JOB_ID3")
echo "    Job response:"
echo "$VIEW3" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=4, ensure_ascii=False))" 2>/dev/null || echo "    $VIEW3"

KS_STATUS=$(echo "$VIEW3" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null)
KS_ERROR=$(echo "$VIEW3" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error','') or '')" 2>/dev/null)

echo ""
if [ "$KS_STATUS" = "failed" ]; then
    echo "$PASS — status=failed (kill switch bloqueó el job)"
else
    echo "$FAIL — status=$KS_STATUS (esperado: failed)"
fi

if echo "$KS_ERROR" | grep -q "killed_by_switch"; then
    echo "$PASS — error contiene 'killed_by_switch': $KS_ERROR"
else
    echo "$FAIL — error NO contiene 'killed_by_switch': '$KS_ERROR'"
fi

# 3e — eliminar PAUSE
sudo rm "$PAUSE_PATH"
if [ ! -f "$PAUSE_PATH" ]; then
    echo "3e. PAUSE eliminado: $PASS"
else
    echo "3e. PAUSE no eliminado: $FAIL"
fi

echo ""
echo "============================================================"
echo "FIN DE PRUEBAS"
echo "============================================================"
