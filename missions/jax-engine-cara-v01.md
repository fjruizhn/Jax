faceta: hyde

# JAX Platform — JAX Engine + LA CARA v0.1

__product__ = "JAX Platform"
__module__ = "JAX Engine + LA CARA"
__version__ = "0.1.0"
__dedication__ = "En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs."

## Contexto
JAX Platform es un producto independiente de AteneaERP.
Vive en ~/jax-platform/ en hall9000.
Backend: FastAPI (Python 3.12, mismo patrón que LAS MANOS).
Frontend: React 19 + Tailwind + Vite + Zustand + WebSockets.
Multi-tenant desde el primer commit — tenant_id/user_id en TODOS los registros.
Fernando es tenant_id=1, user_id=1, role=superadmin (seedado, no administrable todavía).

## Estructura a crear

~/jax-platform/
  backend/
    main.py               — FastAPI app + startup + CORS + WebSocket
    jax_engine/
      __init__.py
      state.py            — Estado vivo del ecosistema (facetas, pipelines, jobs)
      events.py           — EventBus: pub/sub de eventos JSON
      resource_manager.py — Cuotas, concurrencia, admission control
      schemas.py          — Pydantic models para eventos y estado
      websocket_hub.py    — WebSocket manager por usuario
    auth/
      __init__.py
      jwt.py              — JWT encode/decode, access + refresh tokens
      middleware.py       — FastAPI middleware de autenticación
      models.py           — User, Tenant, Session
    api/
      __init__.py
      events.py           — GET /api/events (SSE fallback)
      health.py           — GET /api/health
      auth.py             — POST /api/auth/login, POST /api/auth/refresh
      state.py            — GET /api/state (estado completo del ecosistema)
      pipelines.py        — GET /api/pipelines (proxy a Jacobs en LAS MANOS)
      facets.py           — GET /api/facets (estado de facetas)
    db/
      __init__.py
      connection.py       — MariaDB connection (mismas credenciales JAX_DB_*)
      migrations.py       — CREATE TABLE IF NOT EXISTS para tablas nuevas
      seed.py             — Fernando tenant_id=1, user_id=1, superadmin
    requirements.txt
  frontend/
    package.json
    vite.config.js
    tailwind.config.js
    index.html
    src/
      main.jsx
      App.jsx
      store/
        useJaxStore.js    — Zustand: estado global (facetas, pipeline activo, ojo)
        useWebSocket.js   — Hook WebSocket con reconexión automática
      components/
        HalEye/
          HalEye.jsx      — SVG animado del ojo HAL 9000
          HalEye.css      — Animaciones CSS por estado/faceta
        LeftPanel/
          LeftPanel.jsx   — Lista de facetas con estado y color
          FacetCard.jsx   — Tarjeta por faceta (icono, estado, último mensaje)
        CenterPanel/
          CenterPanel.jsx — Conversación + ojo HAL central
          Message.jsx     — Mensaje con icono de faceta + markdown
        RightPanel/
          RightPanel.jsx  — Jacobs: pipeline activo, steps, botón aprobar
          StepCard.jsx    — Card por step (estado, faceta, duración)
          AuditLog.jsx    — Últimos 20 eventos de LAS MANOS
        BottomBar/
          BottomBar.jsx   — Input + selector modo + voz + kill switch
          KillSwitch.jsx  — Botón rojo visible siempre
        Notifications/
          Toast.jsx       — Notificaciones de pipeline/gate/kill switch
      pages/
        Login.jsx         — Login simple email/password
        Dashboard.jsx     — LA CARA completa (3 paneles + ojo)
      api/
        client.js         — axios con JWT interceptor
        websocket.js      — WebSocket client con reconexión

## JAX Engine — Especificaciones

### state.py
Estado vivo del ecosistema:
- facets: dict[str, FacetState] — estado de cada faceta (idle/thinking/error)
- active_pipelines: dict[str, PipelineState] — pipelines en curso
- las_manos_alive: bool — health check cada 30s a 127.0.0.1:7777/health
- connected_users: dict[str, UserSession] — usuarios con WebSocket activo

### events.py
EventBus con pub/sub:
- publish(event: JAXEvent) — emite evento a todos los suscriptores del tenant
- subscribe(tenant_id, user_id, callback)
- unsubscribe(user_id)

Eventos base (JAXEvent):
{
  "event_id": "uuid",
  "event_type": "facet_status_changed|pipeline_step_changed|human_gate_requested|kill_switch_activated|las_manos_health_changed|facet_response_completed",
  "tenant_id": "1",
  "user_id": "1",
  "payload": {},
  "timestamp": "ISO8601"
}

### resource_manager.py
Admission control (NUNCA interrumpe trabajo en vuelo):
- can_start_pipeline(tenant_id) → bool — verifica límite 3 pipelines
- admit_pipeline(tenant_id, pipeline_id) — registra
- release_pipeline(tenant_id, pipeline_id) — libera al terminar

Rate limiting solo en nuevas invocaciones:
- rejected_by_quota — rechaza nuevo request
- rejected_by_concurrency — rechaza si hay 3 pipelines activos
- NUNCA cancelled_by_rate_limit sobre trabajo en vuelo

### websocket_hub.py
Canal por usuario:
- connect(user_id, websocket)
- disconnect(user_id)
- send_to_user(user_id, event)
- broadcast_to_tenant(tenant_id, event)

WebSocket URL: ws://localhost:8080/ws/{user_id}?token={jwt}

## Auth — Especificaciones

JWT:
- Access token: 15 minutos, HS256
- Refresh token: 7 días, HttpOnly cookie
- Payload: {user_id, tenant_id, role, exp}

Tablas nuevas en jax_memory:
CREATE TABLE IF NOT EXISTS jax_tenants (
  tenant_id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  plan VARCHAR(20) DEFAULT 'personal',
  status VARCHAR(20) DEFAULT 'active',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jax_users (
  user_id INT AUTO_INCREMENT PRIMARY KEY,
  tenant_id INT NOT NULL,
  email VARCHAR(100) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(20) DEFAULT 'operator',
  status VARCHAR(20) DEFAULT 'active',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (tenant_id) REFERENCES jax_tenants(tenant_id)
);

Seed obligatorio:
- tenant_id=1, name="Inversiones Diamante Negro", plan="superadmin"
- user_id=1, email="fernando@rich-hn.com", role="superadmin", password desde JAX_SEED_ADMIN_PASSWORD (o generada aleatoria y logueada al arranque si no está seteada, ver db/seed.py) (hash bcrypt)

## LA CARA — El Ojo HAL 9000

SVG animado con estados visuales:
- reposo: pulso lento azul frío (#3b82f6)
- jax_local thinking: pulso medio azul
- jekyll thinking: índigo (#6366f1)
- hyde thinking: naranja (#f97316)
- hipatia thinking: verde esmeralda (#10b981)
- thot thinking: dorado (#f59e0b)
- kimi thinking: cian eléctrico (#06b6d4)
- ada thinking: violeta (#7c3aed)
- jacobs running: blanco pulsante (#ffffff)
- human_gate_requested: anillo ámbar parpadeante
- kill_switch_active: rojo fijo sin pulso (#ef4444)
- las_manos_down: ojo apagado (gris #374151)

El ojo lee estado del useJaxStore — NO tiene lógica propia.

## Layout de LA CARA

Pantalla completa dividida en:
- Panel izquierdo (20%): lista de facetas
- Panel central (55%): ojo HAL + conversación
- Panel derecho (25%): Jacobs + audit log
- Barra inferior fija: input + controles
- Fondo oscuro (#0f172a) — tema HAL 9000

## Backend — Puerto y configuración

JAX Platform Backend corre en puerto 8080 (distinto a LAS MANOS en 7777).
Frontend en puerto 5173 (dev) o servido por FastAPI en producción.
CORS: permite localhost:5173 y el dominio de producción futuro.

Variables de entorno (leer de /etc/jax/.env):
JAX_DB_HOST, JAX_DB_USER, JAX_DB_PASSWORD, JAX_DB_NAME
JAX_JWT_SECRET (generar si no existe: secrets.token_hex(32))
LAS_MANOS_URL=http://127.0.0.1:7777
JACOBS_URL=http://127.0.0.1:7777/jacobs

## Proxy a LAS MANOS y Jacobs

El backend de JAX Platform actúa como proxy:
- GET /api/pipelines → GET 127.0.0.1:7777/jacobs/pipeline (con tenant_id filter)
- POST /api/pipelines → POST 127.0.0.1:7777/jacobs/pipeline
- POST /api/pipelines/{id}/resume → POST 127.0.0.1:7777/jacobs/pipeline/{id}/resume
- GET /api/facets → lee state.py del JAX Engine
- POST /api/chat → invoca faceta via JAX (futuro)

## Polling desde JAX Engine a LAS MANOS

El JAX Engine hace polling cada 5s a LAS MANOS para:
- GET /health → actualiza las_manos_alive
- GET /jacobs/pipeline/{id} → actualiza estado de pipelines activos
- Cuando hay cambio de estado → publica evento via EventBus

## Servicio systemd

Crear /etc/systemd/system/jax-platform.service:
[Unit]
Description=JAX Platform Backend
After=network.target jax-las-manos.service
Wants=jax-las-manos.service

[Service]
Type=simple
User=fruiz
WorkingDirectory=/home/fruiz/jax-platform/backend
EnvironmentFile=/etc/jax/.env
ExecStart=/home/fruiz/jax-platform/backend/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target

## Frontend — build y servir

En desarrollo: npm run dev (puerto 5173)
En producción: npm run build → FastAPI sirve dist/ en /

## Verificaciones obligatorias

1. GET http://localhost:8080/api/health → {"service":"JAX Platform","status":"alive","las_manos":"alive"}
2. POST http://localhost:8080/api/auth/login con email/password → JWT válido
3. WS ws://localhost:8080/ws/1?token={jwt} → conecta y recibe heartbeat cada 30s
4. GET http://localhost:8080/api/state → estado del ecosistema con facetas y pipelines
5. Frontend: npm run dev → abre en localhost:5173, muestra ojo HAL pulsando
6. Ojo HAL cambia color cuando se simula evento facet_status_changed
7. Panel derecho muestra pipeline activo de Jacobs
8. Kill switch visible en barra inferior

## Dependencias Python (requirements.txt)
fastapi>=0.115
uvicorn>=0.30
httpx>=0.27
pydantic>=2.7
python-jose[cryptography]>=3.3
passlib[bcrypt]>=1.7
aiomysql>=0.2
python-multipart>=0.0.9

## Dependencias Node (package.json)
react: 19
react-dom: 19
vite: 6
tailwindcss: 3
zustand: 5
axios: 1.7
react-markdown: 9
@heroicons/react: 2

## NO tocar
- ~/jax/ (JAX original — producto diferente)
- LAS MANOS (solo leer vía API)
- AteneaERP

## Estructura de carpetas final esperada
~/jax-platform/
  backend/
  frontend/
  README.md

Escribir resultado en ~/jax/missions/jax-engine-cara-v01_result.md

En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.
