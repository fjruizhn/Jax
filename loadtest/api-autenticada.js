// Línea base de los caminos AUTENTICADOS de jax-platform — política 4.
//
// POR QUÉ ESTOS Y NO `/api/chat`. El turno de chat es el camino caro de verdad, pero
// medirlo bajo carga contra producción invoca a Ollama —GPU real— y escribe turnos en la
// memoria: la prueba ensuciaría justo los datos que el sistema usa para recordar. Estos
// tres son de SOLO LECTURA, pasan por el mismo middleware de JWT y tocan la misma base,
// así que miden el camino autenticado sin efectos.
//
// Queda declarado lo que NO se mide acá: el chat con búsqueda semántica y el despacho de
// pipelines. Para esos hay que montar un entorno con su propia base y su propio Ollama,
// y eso es una ronda aparte — no un número que se pueda sacar sin ensuciar producción.
//
// USO:  k6 run -e TOKEN=$(...) loadtest/api-autenticada.js
//
// El token se pasa por entorno y NUNCA se escribe en el archivo.

import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.BASE || 'http://127.0.0.1:8080';
const TOKEN = __ENV.TOKEN;
const VUS = parseInt(__ENV.VUS || '20', 10);

export const options = {
  stages: [
    { duration: '5s', target: VUS },
    { duration: '15s', target: VUS },
    { duration: '5s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
  },
};

const RUTAS = ['/api/pipelines', '/api/motors/capabilities', '/api/facets'];

export default function () {
  const ruta = RUTAS[__ITER % RUTAS.length];
  const r = http.get(`${BASE}${ruta}`, {
    headers: { Authorization: `Bearer ${TOKEN}` },
    tags: { ruta },
  });
  check(r, { 'status 200': (res) => res.status === 200 });
}
