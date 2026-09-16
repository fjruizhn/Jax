// Escenario de carga con GATES automaticos -- politica 4 de LAS CUATRO DEL
// RENDIMIENTO.
//
// QUE APORTA SOBRE scripts/load_test.py. El arnes de stdlib mide y reporta;
// hay que mirar el numero y decidir. k6 agrega dos cosas que hacen de la
// politica un gate y no un informe:
//   - `thresholds`: si el p95 o la tasa de error se pasan del limite, k6
//     termina con exit != 0, asi que un lanzamiento se puede condicionar a eso.
//   - rampas: la carga sube y baja como en la vida real, en vez de disparar N
//     peticiones de golpe. El punto de degradacion aparece solo.
//
// Y MIDE MEJOR, que es la razon de fondo. Mismo endpoint, misma concurrencia
// (50), 2026-09-11:
//     arnes con httpx ....  1.275 rps   p95 100,5 ms
//     arnes con stdlib ...  5.638 rps   p95   6,53 ms
//     k6 ................. 20.399 rps   p95   3,01 ms
// El arnes de Python esta limitado por el GIL: sus hilos no corren en
// paralelo de verdad, asi que a partir de cierta carga lo que se mide es el
// arnes. Para saber cuanto aguanta el servicio, la referencia es k6.
//
// k6 NO reemplaza al arnes de stdlib: en atem-ai (.11) no esta instalado y el
// de stdlib corre en cualquier maquina sin instalar nada. Aquel es para medir
// siempre; este, para medir en serio antes de un lanzamiento.
//
// INSTALACION (hall9000, 2026-09-11): binario oficial de grafana/k6 v2.2.0 en
// ~/bin, con el sha256 verificado contra el checksums.txt de la release. NO se
// uso el snap: lo publica un tercero, no Grafana. No se agrego ningun repo de
// apt al sistema.
//
// USO:  k6 run loadtest/health.js
//       k6 run -e URL=http://127.0.0.1:8080/api/health -e VUS=100 loadtest/health.js
//
// QUE SIGNIFICA EL VERDE, desde el 2026-09-16. Hasta esa fecha /health
// devolvia {"status": "alive"} FIJO: respondia "vivo" por el mero hecho de
// poder responder. Estos numeros median FastAPI devolviendo un literal, no el
// servicio -- con la base caida, el endpoint seguia en 200 y este guion seguia
// en verde. Ahora /health comprueba lo que LAS MANOS necesita para trabajar
// (que la base responda y que el log forense sea escribible) y devuelve 503 si
// algo falta, asi que:
//
//   - un rojo aqui puede significar DOS cosas distintas: que el servicio se
//     degrada bajo carga, o que una dependencia esta caida desde antes de
//     empezar. El cuerpo de la respuesta dice cual, en "problemas".
//   - las comprobaciones van cacheadas 5 s (_SALUD_TTL_SEGUNDOS en server.py),
//     asi que 20.000 req/s NO son 20.000 consultas: son una cada 5 s. Si ese
//     TTL sube mucho, este guion deja de ver una caida que ocurra durante la
//     meseta de 20 s.
//
// Los limites de abajo se midieron contra el endpoint VIEJO (2026-09-11). El
// camino nuevo agrega un dict y, una vez cada 5 s, una consulta -- pendiente
// volver a medir para confirmar que p95 < 50 ms sigue holgado.
//
// ADVERTENCIA: contra produccion, esto ES trafico de produccion.

import http from 'k6/http';
import { check } from 'k6';

const URL = __ENV.URL || 'http://127.0.0.1:8080/api/health';
const VUS = parseInt(__ENV.VUS || '50', 10);

export const options = {
  stages: [
    { duration: '10s', target: VUS },       // rampa de subida
    { duration: '20s', target: VUS },       // meseta: aca se mide
    { duration: '5s', target: 0 },          // bajada
  ],
  thresholds: {
    // Los limites salen de la corrida real del 2026-09-11 (p95 3,01 ms a 50
    // VUs, cero errores), con holgura para que un pico normal no de rojo. Si
    // el servicio empeora de verdad, esto falla y el exit code lo dice.
    http_req_duration: ['p(95)<50'],
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
  },
};

export default function () {
  const r = http.get(URL);
  check(r, {
    'status 200': (res) => res.status === 200,
    'sin 5xx': (res) => res.status < 500,
  });
}
