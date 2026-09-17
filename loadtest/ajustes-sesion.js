// Carga del frente C (2026-09-16, jax-platform "ajustes que mandan") -- política 4.
//
// Mide los dos caminos de sesión que ahora LEEN un ajuste por request:
//   login   -> lee session_timeout_min antes del bcrypt y emite la cookie con esa vida
//   refresh -> lee session_timeout_min y mide la vida contra `iat`
// Contra una instancia AISLADA (base jax_memory_test, sin tareas de fondo), nunca
// contra producción: el login sube token_version (sesión única) y mataría sesiones reales.
//
// USO:
//   k6 run -e BASE=http://127.0.0.1:18380 -e ESCENARIO=login   -e EMAIL=... -e PASSWORD=... loadtest/ajustes-sesion.js
//   k6 run -e BASE=http://127.0.0.1:18380 -e ESCENARIO=refresh -e REFRESH=... loadtest/ajustes-sesion.js
// Credenciales y token por entorno; NUNCA se escriben en el archivo.

import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.BASE || 'http://127.0.0.1:18380';
const ESCENARIO = __ENV.ESCENARIO;
const VUS = parseInt(__ENV.VUS || (ESCENARIO === 'login' ? '10' : '25'), 10);

if (ESCENARIO !== 'login' && ESCENARIO !== 'refresh') {
  throw new Error('ESCENARIO tiene que ser login o refresh');
}

export const options = {
  stages: [
    { duration: '5s', target: VUS },
    { duration: '20s', target: VUS },
    { duration: '5s', target: 0 },
  ],
  thresholds: {
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
  },
};

export default function () {
  if (ESCENARIO === 'login') {
    const r = http.post(`${BASE}/api/auth/login`,
      JSON.stringify({ email: __ENV.EMAIL, password: __ENV.PASSWORD }),
      { headers: { 'Content-Type': 'application/json' }, tags: { escenario: 'login' } });
    check(r, { 'login 200': (res) => res.status === 200 });
  } else {
    const r = http.post(`${BASE}/api/auth/refresh`, null,
      { headers: { Cookie: `refresh_token=${__ENV.REFRESH}` }, tags: { escenario: 'refresh' } });
    check(r, { 'refresh 200': (res) => res.status === 200 });
  }
}
