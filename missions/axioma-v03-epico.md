faceta: hyde

# Axioma v0.3 — La misión épica nocturna

## CONTEXTO OBLIGATORIO
Leer PRIMERO: ~/.claude/CLAUDE.md y ~/jax-platform/CLAUDE.md
Políticas absolutas: i18n ES/EN, dark/light mode, sin hardcoding, backup antes de modificar.
Stack: React 19 + Tailwind + Zustand + react-i18next (frontend) / FastAPI + Python 3.12 (backend)

---

## TAREA 1 — Fix imagen empuja botones

El chat no tiene scroll interno y las imágenes grandes empujan la BottomBar fuera de pantalla.

Fixes:
- Panel central del chat: overflow-y auto, height calculado para no empujar BottomBar
- Imágenes en el chat: max-height 400px, object-fit contain, border-radius 8px
- BottomBar: position sticky bottom-0 o fixed, siempre visible
- Verificar en dark y light mode

---

## TAREA 2 — Adjuntar archivos al chat

Botón "+" junto al input del chat en TODOS los modos (Chat, Comando, Pipeline, Imagen).

### Tipos de archivo soportados:
- Imágenes: JPG, PNG, GIF, WebP, SVG → mostrar preview inline
- PDF → extraer texto y enviarlo como contexto a la faceta
- Texto: TXT, MD, CSV, JSON, TOML → enviar contenido como contexto
- Código: PY, JS, JSX, TS, HTML, CSS → enviar con syntax highlighting

### Frontend:
- Botón "+" abre selector de archivo (input type=file, accept múltiple)
- Preview del archivo seleccionado sobre el input (nombre + ícono + botón X para quitar)
- Al enviar: archivo se adjunta al mensaje
- Imágenes → mostrar inline en el chat
- Otros → mostrar como adjunto con ícono y nombre

### Backend: POST /api/chat/upload
- Recibe archivo multipart/form-data
- Imágenes → base64 → enviar a la faceta que soporte visión (Hipatia/Gemini, Thot/GPT)
- PDF → extraer texto con pypdf2 o pdfplumber → enviar como texto
- Texto/código → enviar directo como contexto
- Devolver: {file_id, type, content_preview, ready: true}

### Facetas con visión:
- Hipatia (Gemini 2.5 Flash) → soporta imágenes nativo
- Thot (GPT-5.5) → soporta imágenes nativo
- Hyde (modo Comando) → puede leer cualquier archivo
- Las demás → solo texto extraído

### i18n: todos los strings en es.js y en.js
### dark/light: el preview y adjuntos funcionan en ambos modos

---

## TAREA 3 — Módulo de Administración completo

Ruta: /admin (protegida, solo role=superadmin)
Navbar superior con enlace "Admin" visible solo para superadmin.

### 3A — Dashboard Admin
- Cards con estado de todos los servicios:
  - LAS MANOS (:7777) — alive/down + uptime
  - JAX Engine (:8080) — alive/down
  - Frontend (:5173) — alive
  - MariaDB — connected/error
- Estadísticas del día: mensajes enviados, pipelines completados, imágenes generadas
- Últimos 5 eventos del audit log

### 3B — Gestión de API Keys
Tabla: Proveedor | Modelo | Key (últimos 4 chars) | Estado | Última prueba | Acción

Providers:
- OpenAI → Thot (GPT-5.5) + DALL-E (gpt-image-1)
- DeepSeek → Jekyll (V4 Flash)
- Gemini → Hipatia (2.5 Flash)
- Moonshot → Kimi (K2.7 Code)
- Z.ai → Ada (GLM-5.2) — pendiente

Acciones por key:
- "Probar conexión" → ping a la API → OK/Error con latencia
- "Rotar key" → modal para ingresar nueva key → escribe en /etc/jax/.env
- "Activar/Desactivar" → habilita o deshabilita la faceta

Backend:
- GET /api/admin/keys → lista providers con estado (NUNCA devolver la key completa)
- POST /api/admin/keys/{provider}/test → hace ping a la API del provider
- PUT /api/admin/keys/{provider} → actualiza la key en /etc/jax/.env (reload del servicio)

### 3C — Gestión de Usuarios
Tabla: Email | Rol | Estado | Último acceso | Acciones

Roles: superadmin, operator, viewer
Acciones:
- Crear usuario (email + rol + password temporal)
- Cambiar rol
- Activar/Desactivar
- Reset password

Backend CRUD sobre tabla jax_users.
Emails de bienvenida y reset via la configuración de mail existente.

### 3D — Repositorio de Artefactos
Vista de archivos para ~/jax/repo/
Estructura visual en árbol:
  missions/ — misiones guardadas
  pipelines/ — resultados de pipelines  
  documents/ — documentos (BIBox concepto, etc.)
  images/ — imágenes generadas

Funcionalidades:
- Listar archivos con nombre, fecha, tamaño
- Preview de contenido al hacer click (texto/markdown/imagen)
- Botón descargar
- Botón eliminar (con confirmación)
- Botón "Guardar aquí" desde resultado de Hyde o pipeline completado

Backend:
- GET /api/admin/repo → lista archivos por carpeta
- GET /api/admin/repo/file?path=... → contenido del archivo
- DELETE /api/admin/repo/file?path=... → eliminar
- POST /api/admin/repo/save → guardar resultado de tarea en repo

### 3E — Configuración del Sistema
Formulario con:
- Idioma por defecto (ES/EN)
- Tema por defecto (dark/light)
- Timeout de sesión JWT (minutos)
- Max pipelines simultáneos (1-3)
- Retención de web-tasks (días, default 7)
- Notificaciones WS (on/off por tipo)
- Nombre del sistema (default: "Axioma")
- Logo personalizado (upload)

Guardar en tabla axioma_config (key/value).
Backend: GET/PUT /api/admin/config

### 3F — Monitor de Costos
Tabla por faceta:
- Tokens entrada + salida (hoy / semana / mes)
- Costo estimado USD (usando precios hardcodeados por modelo)
- Requests totales

Precios por modelo (hardcoded en config):
- gpt-5.5: $5.00/M in, $30.00/M out (Thot)
- deepseek-v4-flash: $0.14/M in, $0.28/M out (Jekyll)
- gemini-2.5-flash: $0.15/M in, $0.60/M out (Hipatia)
- kimi-k2.7-code: $0.95/M in, $4.00/M out (Kimi)
- gpt-image-1: $0.04/imagen (DALL-E)

Gráfico de barras: uso por faceta últimos 7 días (usar Chart.js o recharts).

Guardar uso en tabla axioma_usage.
Registrar cada request en el backend al procesar chat/imagen/pipeline.

---

## TABLAS DB NUEVAS

```sql
CREATE TABLE IF NOT EXISTS axioma_config (
  config_key VARCHAR(100) PRIMARY KEY,
  config_value TEXT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS axioma_usage (
  id INT AUTO_INCREMENT PRIMARY KEY,
  tenant_id INT DEFAULT 1,
  user_id INT DEFAULT 1,
  facet VARCHAR(30) NOT NULL,
  model VARCHAR(50) NOT NULL,
  tokens_in INT DEFAULT 0,
  tokens_out INT DEFAULT 0,
  cost_usd DECIMAL(10,6) DEFAULT 0,
  request_type VARCHAR(20) DEFAULT 'chat',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS axioma_artifacts (
  id INT AUTO_INCREMENT PRIMARY KEY,
  tenant_id INT DEFAULT 1,
  user_id INT DEFAULT 1,
  name VARCHAR(200) NOT NULL,
  artifact_type VARCHAR(30) NOT NULL,
  file_path TEXT NOT NULL,
  size_bytes INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## ESTRUCTURA DE ARCHIVOS NUEVA
frontend/src/

pages/

Admin.jsx              — layout admin con sidebar

admin/

AdminDashboard.jsx

AdminApiKeys.jsx

AdminUsers.jsx

AdminRepository.jsx

AdminSettings.jsx

AdminCosts.jsx

components/

admin/

AdminSidebar.jsx

ServiceStatusCard.jsx

ApiKeyRow.jsx

UserRow.jsx

FileTree.jsx

CostChart.jsx

chat/

FileAttachment.jsx   — preview de archivo adjunto

AttachButton.jsx     — botón "+"
backend/api/

admin/

init.py

dashboard.py

keys.py

users.py

repository.py

config.py

usage.py

upload.py               — manejo de archivos adjuntos

---

## VERIFICACIONES OBLIGATORIAS

1. Fix imagen: chat tiene scroll, BottomBar siempre visible
2. Adjuntar imagen → Hipatia la describe correctamente
3. Adjuntar PDF → texto extraído enviado como contexto
4. /admin → solo visible para superadmin
5. /admin/keys → tabla con todos los providers, botón "Probar" funciona
6. /admin/users → crear usuario, cambiar rol
7. /admin/repo → ver ~/jax/repo/documents/bibox-concepto.md
8. /admin/config → cambiar nombre sistema, guardar en DB
9. /admin/costs → tabla con uso del día (aunque sea 0)
10. py_compile backend, npm run build frontend — sin errores
11. TODO en i18n ES/EN
12. TODO funciona en dark Y light mode
13. Historial de chat persiste al recargar

## DEPENDENCIAS ADICIONALES si faltan
- pdfplumber o pypdf2 (pip install pdfplumber --break-system-packages)
- python-multipart (pip install python-multipart --break-system-packages)

## NO TOCAR
- ~/jax/ (JAX terminal original)
- LAS MANOS internals
- Jacobs internals
- Motor Registry internals

Escribir resultado detallado en ~/jax/missions/axioma-v03-epico_result.md

En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.
Construido desde Honduras para el mundo.
