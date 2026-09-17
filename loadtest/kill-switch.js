// Kill switch bajo carga, EN PRODUCCIÓN y SOLO CON EL FRENO PUESTO (frente B,
// 2026-09-16). LAS CUATRO DEL RENDIMIENTO, política 4.
//
// POR QUÉ ES SEGURO. El escenario `freno` manda un chat a una faceta que NO
// existe: con el freno puesto responde 423 antes de todo; si el freno no
// estuviera puesto, respondería 400 por faceta desconocida (api/chat.py
// valida antes de la memoria y del modelo). Nunca llega a un modelo. Además,
// setup() aborta si el freno no está puesto. `activar_idempotente` pide
// activar lo que ya está activo: 200 con cambio=false, sin fila de auditoría.
//
// USO (el TOKEN es un access token de superadmin que da Fernando para esta
// ventana; vence en 15 min; nunca se escribe en un archivo):
//   k6 run -e TOKEN=... loadtest/kill-switch.js

import http from 'k6/http';
import { check } from 'k6';
import exec from 'k6/execution';

const BASE = __ENV.BASE || 'http://127.0.0.1:8080';
const TOKEN = __ENV.TOKEN;
const VUS = parseInt(__ENV.VUS || '50', 10);
const CABECERAS = { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' };

http.setResponseCallback(http.expectedStatuses(200, 423));

const rampa = (vus) => [
  { duration: '5s', target: vus },
  { duration: '20s', target: vus },
  { duration: '5s', target: 0 },
];

export const options = {
  scenarios: {
    freno: { executor: 'ramping-vus', exec: 'freno', stages: rampa(VUS) },
    admin: { executor: 'ramping-vus', exec: 'admin', stages: rampa(Math.max(1, Math.floor(VUS / 5))) },
    activar_idempotente: { executor: 'ramping-vus', exec: 'activarIdempotente', stages: rampa(10) },
  },
  thresholds: {
    'http_req_duration{escenario:freno}': ['p(95)<500'],
    'http_req_duration{escenario:admin}': ['p(95)<500'],
    'http_req_duration{escenario:activar}': ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
  },
};

export function setup() {
  if (!TOKEN) exec.test.abort('falta TOKEN');
  const r = http.get(`${BASE}/api/admin/kill-switch`, { headers: CABECERAS });
  if (r.status !== 200 || r.json('activo') !== true) {
    exec.test.abort(`el freno NO está puesto (status ${r.status}): esta carga sólo corre con el kill switch activo`);
  }
  // Informativo (Task H, 2026-09-17): si el freno viene de la ruta heredada
  // (la vieja, que el módulo interruptor sigue leyendo), se dice. No aborta ni cuenta como fallo: el check sólo
  // exige que el campo exista.
  const heredada = r.json('heredada');
  check(r, { 'admin informa heredada': () => typeof heredada === 'boolean' });
  console.log(`freno puesto; heredada=${heredada}`);
}

export function freno() {
  const r = http.post(`${BASE}/api/chat`, JSON.stringify({ message: 'carga-kill-switch', facet: '__carga_kill_switch__' }),
    { headers: CABECERAS, tags: { escenario: 'freno' } });
  check(r, { '423 kill_switch_activo': (x) => x.status === 423 && x.json('detail') === 'kill_switch_activo' });
}

export function admin() {
  const r = http.get(`${BASE}/api/admin/kill-switch`, { headers: CABECERAS, tags: { escenario: 'admin' } });
  check(r, { 'admin: activo': (x) => x.status === 200 && x.json('activo') === true });
  const s = http.get(`${BASE}/api/state`, { headers: CABECERAS, tags: { escenario: 'admin' } });
  check(s, { 'state: kill_switch_active': (x) => x.status === 200 && x.json('kill_switch_active') === true });
}

export function activarIdempotente() {
  const r = http.post(`${BASE}/api/admin/kill-switch/activar`, null, { headers: CABECERAS, tags: { escenario: 'activar' } });
  check(r, { 'activar sin cambio': (x) => x.status === 200 && x.json('cambio') === false });
}
