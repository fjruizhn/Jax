# BIBox — Business Intelligence Box
## Concepto inicial · 19 de junio de 2026

> "No perdemos nada de lo que hicimos hoy." — Fernando Ruiz

---

## La idea

Un dispositivo físico que llega a la empresa con todo preinstalado:
servidor, ERP, AI, mail, backups — sin dependencia de nube,
sin mensualidades por usuario, con soberanía total de los datos.

El cliente lo conecta a su red, entra a un wizard,
configura su empresa en 30 minutos, y tiene su sistema listo.

---

## Hardware propuesto

**MINISFORUM N5 MAX** (Ryzen AI Max+ 395, 16C/32T)
- 64GB LPDDR5X unificada (CPU+GPU+NPU comparten el pool)
- 126 TOPS de AI
- 2x 10GbE, 2x USB4 V2, 5x bahías HDD, 5x M.2
- Capacidad hasta 200TB
- Factor de forma: NAS de escritorio, ~8x8x10 pulgadas

**Configuración de discos:**
- M.2 #1 (2TB NVMe): OS Ubuntu 24.04 + JAX/Axioma + sistema
- M.2 #2 (2TB NVMe): VMs (ERP + TrueNAS)
- HDD #1 (6-8TB): MariaDB del ERP — solo para datos
- HDD #2 (6-8TB): Backups TrueNAS
- HDD #3 (opcional): Expansión futura

---

## Arquitectura de software

### Host/Matriz (Ubuntu 24.04 LTS)
- JAX con todas las facetas (JAX Local, Jekyll, Hyde, Hipatia, Thot, Kimi, Ada)
- Axioma Platform (cabina de mando del sistema)
- KVM hypervisor (gestiona las VMs)
- Servicios: jax-las-manos (:7777), jax-platform (:8080), jax-frontend (:5173)

### VM 1: BizServer (aaPanel PRO)
- Nginx + PHP 8.3
- AteneaERP (Laravel 13)
- Servidor de correo completo (Postfix + Dovecot + Roundcube)
- MariaDB 11.x (apunta al HDD dedicado)
- SSL automático (Let's Encrypt o self-signed para red local)

### VM 2: BizNAS (TrueNAS SCALE)
- Gestiona HDD de backups
- Restic automático de todo el sistema
- Snapshots ZFS
- Acceso SMB/NFS para la empresa

---

## Wizard de configuración inicial

El cliente conecta el BIBox a su red y accede a:
**http://bibox.local** o **http://[IP]**

**Paso 1 — Empresa**
- Nombre de la empresa
- País, zona horaria, moneda
- Logo (opcional)

**Paso 2 — Dominio y correo**
- Dominio propio (ej: empresa.com) o subdominio BIBox
- Configuración de mail server
- Certificado SSL

**Paso 3 — Administrador**
- Email del admin
- Password
- Módulos a activar

**Paso 4 — AI**
- Activar JAX/Axioma
- Seleccionar facetas disponibles
- API keys propias o servicio incluido

**Paso 5 — Confirmación**
- BIBox configura todo automáticamente
- 10-15 minutos de instalación
- Dashboard listo

---

## Propuesta de valor

**Para la empresa:**
- Sin mensualidades de nube por usuario
- Sus datos físicamente en sus instalaciones
- ERP + AI + Mail + Backups en un solo dispositivo
- Soberanía total
- Sin dependencia de internet para operación interna

**Para Axioma (el negocio):**
- Hardware: margen de $1,500-2,000 por unidad
- Licencia AteneaERP: $1,200-2,400/año
- Soporte: $600-1,200/año
- Revenue recurrente sin infraestructura de nube

---

## Estimado de costos y precio

| Componente | Costo estimado |
|---|---|
| N5 MAX diskless | ~$2,899 |
| 2x M.2 2TB Samsung 990 Pro | ~$300 |
| 2x HDD 6TB | ~$200 |
| Ensamble + configuración + pruebas | ~$500 |
| **Costo total hardware** | **~$3,900** |
| Licencia AteneaERP año 1 | $1,200 |
| Implementación + capacitación | $800 |
| **Precio al cliente (año 1)** | **$7,000-8,500** |
| **Renovación anual** | **$1,800-2,400** |

---

## Mercado objetivo inicial

- PyMEs Honduras y LATAM con 5-50 empleados
- Empresas que NO quieren datos en la nube
- Sectores: comercio, manufactura, servicios profesionales
- Bancos y cooperativas pequeñas (versión con HAMMURABI)

---

## Lo que falta para el v1

### Ya existe:
- ✅ AteneaERP (multi-tenant, roles, módulos)
- ✅ Axioma Platform (JAX + interfaz)
- ✅ LAS MANOS + Jacobs + Motor Registry
- ✅ Arquitectura multi-tenant desde commit 1

### Por construir:
- ⏳ Wizard de instalación (React — parte de Axioma)
- ⏳ BIBox installer script (bash — automatiza el setup completo)
- ⏳ Imagen ISO preconfigurada
- ⏳ Portal de licencias y activación
- ⏳ Versión HAMMURABI (para bancos/cooperativas)
- ⏳ Documentación de usuario final

---

## Versiones del producto

**BIBox Starter** — PyMEs pequeñas
- N5 MAX 64GB + 2x M.2 2TB + 2x HDD 4TB
- AteneaERP módulos básicos
- JAX con facetas locales (sin API externas)
- ~$5,500

**BIBox Pro** — PyMEs medianas
- N5 MAX 64GB + 2x M.2 2TB + 3x HDD 8TB
- AteneaERP completo + HAMMURABI básico
- JAX completo con todas las facetas
- ~$8,000

**BIBox Enterprise** — Bancos y cooperativas
- N5 MAX 64GB o hardware personalizado
- AteneaERP + HAMMURABI completo
- JAX con Red Queen (inferencia local de 70B+)
- Precio a convenir

---

## Nombre comercial

**BIBox** — Business Intelligence Box

Tagline: *"Tu empresa. Tus datos. Tu AI."*
O: *"Todo lo que tu empresa necesita. En una caja."*

---

## Nota

Este es un concepto inicial capturado el 19 de junio de 2026.
No se ha tomado ninguna decisión de inversión todavía.
El hardware existe y está disponible. El software existe en ~70%.
La idea es sólida y el mercado en Honduras/LATAM es real.

*En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.*
*Construido desde Honduras para el mundo.*
