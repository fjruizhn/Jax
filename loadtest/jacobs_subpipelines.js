// Carga de POST /jacobs/pipeline con el contrato de sub-pipelines -- politica 4
// de LAS CUATRO DEL RENDIMIENTO (frente F, 2026-09-16).
//
// Contra la app AISLADA loadtest/jacobs_subpipelines_app.py (jax_memory_test),
// nunca contra LAS MANOS de produccion.
//
// ESCENARIO:
//   base      -- jax_local sin token (el "antes", mismo plan sustituto)
//   legitimo  -- ada con un token sembrado distinto por iteracion
//   inventado -- ada con tokens inventados: el peor caso del atacante (rechazo
//                con diagnostico y un evento escrito por pedido)
//
// 422 "pipelines activos" es el limite duro de 3 (MAX_PARALLEL_PIPELINES)
// haciendo su trabajo bajo concurrencia: se cuenta aparte, no como error.
//
// USO:
//   k6 run -e ESCENARIO=legitimo -e VUS=25 -e TOKENS=$S/tokens.json \
//          -e PADRE=$(cat $S/tokens.json.padre) loadtest/jacobs_subpipelines.js

import http from 'k6/http';
import { check } from 'k6';
import exec from 'k6/execution';
import { Counter } from 'k6/metrics';
import { SharedArray } from 'k6/data';

const BASE = __ENV.BASE || 'http://127.0.0.1:7799';
const ESCENARIO = __ENV.ESCENARIO || 'legitimo';
const VUS = parseInt(__ENV.VUS || '25', 10);
const PADRE = __ENV.PADRE || '';

const TOKENS = ESCENARIO === 'legitimo'
  ? new SharedArray('tokens', () => JSON.parse(open(__ENV.TOKENS)))
  : [];

const aceptados = new Counter('aceptados');
const limiteParalelo = new Counter('limite_paralelo');
const rechazosToken = new Counter('rechazos_token');

http.setResponseCallback(http.expectedStatuses(200, 403, 422));

export const options = {
  stages: [
    { duration: '5s', target: VUS },
    { duration: '20s', target: VUS },
    { duration: '5s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
  },
};

function cuerpo() {
  const base = { name: 'carga-subpipeline', objective: 'o', mode: 'dry_run' };
  if (ESCENARIO === 'base') {
    return { ...base, invoked_by: 'jax_local' };
  }
  const i = exec.scenario.iterationInTest;
  if (ESCENARIO === 'inventado') {
    return { ...base, invoked_by: 'ada', parent_pipeline_id: PADRE,
             subpipeline_token: `inventado-${exec.vu.idInTest}-${i}` };
  }
  if (i >= TOKENS.length) {
    exec.test.abort(`tokens agotados en la iteracion ${i}: sembrar mas`);
  }
  return { ...base, invoked_by: 'ada', parent_pipeline_id: PADRE, subpipeline_token: TOKENS[i] };
}

export default function () {
  const r = http.post(`${BASE}/jacobs/pipeline`, JSON.stringify(cuerpo()), {
    headers: { 'Content-Type': 'application/json' },
    tags: { escenario: ESCENARIO },
  });
  const limite = r.status === 422 && String(r.body).includes('pipelines activos');
  if (r.status === 200) aceptados.add(1);
  else if (limite) limiteParalelo.add(1);
  else if (r.status === 403) rechazosToken.add(1);
  check(r, {
    // El 422 del limite paralelo se cuenta aparte (limite_paralelo) en los TRES
    // escenarios: no es un fallo del camino del token.
    'respuesta esperada': (res) => (ESCENARIO === 'inventado'
      ? res.status === 403 || limite
      : res.status === 200 || limite),
  });
}
