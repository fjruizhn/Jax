// Carga de POST /jacobs/pipeline con el contrato de sub-pipelines -- politica 4
// de LAS CUATRO DEL RENDIMIENTO (frente F, 2026-09-16).
//
// Contra la app AISLADA loadtest/jacobs_subpipelines_app.py (jax_memory_test),
// nunca contra LAS MANOS de produccion.
//
// ESCENARIO:
//   base      -- la MESA: invoked_by='plataforma', sin token
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
// Credencial de servicio (las_manos/auth_servicio.py): sin ella LAS MANOS responde 401.
// Una credencial POR IDENTIDAD (2026-09-17): auth_servicio solo deja a
// `plataforma` declarar invoked_by='plataforma' y a `jacobs` declarar
// invoked_by='ada'. Con una sola credencial el arnes medía 403 a 28.000/s --
// un 403 no toca la base ni el cupo, o sea que medía FastAPI, no el servicio.
const CREDENCIAL = __ENV.CREDENCIAL || '';
const CREDENCIAL_JACOBS = __ENV.CREDENCIAL_JACOBS || '';
if (!CREDENCIAL) throw new Error('falta CREDENCIAL (JAX_LAS_MANOS_CREDENCIAL_PLATAFORMA de la app de carga)');
if (!CREDENCIAL_JACOBS) throw new Error('falta CREDENCIAL_JACOBS (JAX_LAS_MANOS_CREDENCIAL_JACOBS de la app de carga)');
const PADRE = __ENV.PADRE || '';
// OFFSET (2026-09-17): un token se quema una sola vez. Varias corridas contra
// el MISMO archivo sembrado tienen que arrancar donde terminó la anterior; si
// no, la segunda mide 403 de token ya usado y no el camino que se quería medir.
const OFFSET = parseInt(__ENV.OFFSET || '0', 10);

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
    return { ...base, invoked_by: 'plataforma' };
  }
  const i = exec.scenario.iterationInTest;
  if (ESCENARIO === 'inventado') {
    return { ...base, invoked_by: 'ada', parent_pipeline_id: PADRE,
             subpipeline_token: `inventado-${exec.vu.idInTest}-${i}` };
  }
  if (OFFSET + i >= TOKENS.length) {
    exec.test.abort(`tokens agotados en la iteracion ${OFFSET + i}: sembrar mas`);
  }
  return { ...base, invoked_by: 'ada', parent_pipeline_id: PADRE,
           subpipeline_token: TOKENS[OFFSET + i] };
}

export default function () {
  const credencial = ESCENARIO === 'base' ? CREDENCIAL : CREDENCIAL_JACOBS;
  const r = http.post(`${BASE}/jacobs/pipeline`, JSON.stringify(cuerpo()), {
    headers: { 'Content-Type': 'application/json', 'X-Jax-Credencial-Servicio': credencial },
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
