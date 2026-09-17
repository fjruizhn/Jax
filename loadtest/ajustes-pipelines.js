// Carga del frente C (2026-09-16) -- política 4. POST /api/pipelines lee
// max_pipelines por request ANTES de decidir el cupo. Contra una instancia
// AISLADA con un Jacobs falso (responde al instante): el Jacobs real llama a
// un LLM para planear (20-40 s, GPU) y no se pone bajo carga.
// Con el cupo lleno, la mayoría de las respuestas son 429: ese es el camino
// que lee el ajuste y decide, y es el que se mide. Los 200 son las admisiones.
//
// USO: k6 run -e BASE=http://127.0.0.1:18380 -e TOKEN=... loadtest/ajustes-pipelines.js

import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.BASE || 'http://127.0.0.1:18380';
const VUS = parseInt(__ENV.VUS || '25', 10);

export const options = {
  stages: [
    { duration: '5s', target: VUS },
    { duration: '20s', target: VUS },
    { duration: '5s', target: 0 },
  ],
  thresholds: {
    checks: ['rate>0.99'],
  },
};

export default function () {
  const r = http.post(`${BASE}/api/pipelines`,
    JSON.stringify({ name: 'carga', objective: 'carga', mode: 'dry_run' }),
    {
      headers: { Authorization: `Bearer ${__ENV.TOKEN}`, 'Content-Type': 'application/json' },
      responseCallback: http.expectedStatuses(200, 429),
      tags: { escenario: 'pipelines' },
    });
  check(r, { '200 o 429': (res) => res.status === 200 || res.status === 429 });
}
