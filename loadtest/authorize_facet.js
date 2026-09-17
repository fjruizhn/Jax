// Carga de POST /motor/authorize-facet -- politica 4 de LAS CUATRO DEL
// RENDIMIENTO (2026-09-17, pool de conexiones de Jacobs).
//
// Contra la app AISLADA loadtest/authorize_facet_app.py (jax_memory_test),
// nunca contra LAS MANOS de produccion. Es el camino que jax-platform recorre
// antes de CADA turno de la Mesa web a un facet HTTP: una consulta a
// `facet.allowed_callers` por pedido.
//
// Alterna un caller autorizado (200 allowed=true) y uno que no (200
// allowed=false): los dos caminos leen la misma fila, los dos cuentan.
//
// USO:  k6 run -e VUS=25 loadtest/authorize_facet.js

import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.BASE || 'http://127.0.0.1:7798';
const VUS = parseInt(__ENV.VUS || '25', 10);

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

export default function () {
  const autorizado = __ITER % 2 === 0;
  const caller = autorizado ? 'jax_platform_chat' : 'caller_de_carga_no_autorizado';
  const r = http.post(`${BASE}/motor/authorize-facet`,
    JSON.stringify({ caller, facet: 'hipatia' }),
    { headers: { 'Content-Type': 'application/json' } });
  check(r, {
    'veredicto correcto': (res) => res.status === 200 && res.json('allowed') === autorizado,
  });
}
