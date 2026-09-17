// Carga de POST /motor/authorize-facet -- politica 4 de LAS CUATRO DEL
// RENDIMIENTO (2026-09-17, pool de conexiones de Jacobs).
//
// Contra la app AISLADA loadtest/authorize_facet_app.py (jax_memory_test),
// nunca contra LAS MANOS de produccion. Es el camino que jax-platform recorre
// antes de CADA turno de la Mesa web a un facet HTTP: una consulta a
// `facet.allowed_callers` por pedido.
//
// Rota tres caminos: facet permitido (200 allowed=true), facet sin
// allowed_callers (200 allowed=false, fail-closed) y un caller que la
// credencial de la plataforma no puede declarar (403 del middleware de
// auth_servicio, sin llegar a la base). Desde 2026-09-17 el caller ajeno ya no
// llega a la ruta: lo corta el middleware.
//
// USO:  CREDENCIAL=<JAX_LAS_MANOS_CREDENCIAL_PLATAFORMA de la app> k6 run -e VUS=25 loadtest/authorize_facet.js

import http from 'k6/http';
import { check } from 'k6';

// El 403 del caller ajeno es la respuesta correcta, no una falla del servicio.
http.setResponseCallback(http.expectedStatuses(200, 403));

const BASE = __ENV.BASE || 'http://127.0.0.1:7798';
const VUS = parseInt(__ENV.VUS || '25', 10);
// Credencial de servicio (las_manos/auth_servicio.py): sin ella LAS MANOS responde 401.
const CREDENCIAL = __ENV.CREDENCIAL || '';
if (!CREDENCIAL) throw new Error('falta CREDENCIAL (JAX_LAS_MANOS_CREDENCIAL_* de la app de carga)');

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
  const camino = __ITER % 3;
  const caller = camino === 2 ? 'caller_de_carga_no_autorizado' : 'jax_platform_chat';
  const facet = camino === 1 ? 'hyde' : 'hipatia';
  const r = http.post(`${BASE}/motor/authorize-facet`,
    JSON.stringify({ caller, facet }),
    { headers: { 'Content-Type': 'application/json', 'X-Jax-Credencial-Servicio': CREDENCIAL } });
  check(r, {
    'veredicto correcto': (res) => (camino === 2
      ? res.status === 403
      : res.status === 200 && res.json('allowed') === (camino === 0)),
  });
}
