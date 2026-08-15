-- ============================================================
--  JAX 2.0 — Memoria Persistente
--  MariaDB 11.8+ (VECTOR nativo). Vive en hall9000.
--
--  Memoria hibrida: estructurada (SQL exacto) + semantica
--  (columna VECTOR preparada en facts y messages, se llena
--  en Fase 2 con embeddings de nomic-embed-text via Ollama).
--
--  En memoria de Jairo Urbina.
-- ============================================================

-- Base con utf8mb4 (emojis, modismos, acentos)
CREATE DATABASE IF NOT EXISTS jax_memory
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE jax_memory;

-- ------------------------------------------------------------
-- 1. CONVERSATIONS — raiz de cada sesion de dialogo
-- ------------------------------------------------------------
CREATE TABLE conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    conversation_uuid CHAR(36) NOT NULL UNIQUE,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP NULL,
    total_turns INT DEFAULT 0,
    source VARCHAR(20) DEFAULT 'terminal',     -- terminal/telegram/voz
    memory_processed BOOLEAN DEFAULT FALSE,     -- worker de extraccion ya la proceso?
    memory_processed_at TIMESTAMP NULL,
    INDEX idx_started (started_at),
    INDEX idx_processed (memory_processed)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 2. MESSAGES — cada turno. embedding preparado (Fase 2).
-- ------------------------------------------------------------
CREATE TABLE messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT NOT NULL,
    turn_number INT NOT NULL,
    role ENUM('user','jax_local','jekyll','hyde','hipatia') NOT NULL,
    content TEXT NOT NULL,
    facet_used VARCHAR(20),
    model VARCHAR(50),
    latency_ms INT,
    embedding VECTOR(768) NULL,                 -- nomic-embed-text = 768 dims
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    INDEX idx_conversation (conversation_id),
    INDEX idx_role (role),
    INDEX idx_created (created_at)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 3. FACTS — hechos destilados. Busqueda exacta (FULLTEXT) +
--    semantica (VECTOR). El corazon de la memoria.
-- ------------------------------------------------------------
CREATE TABLE facts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fact_uuid CHAR(36) NOT NULL UNIQUE,
    fact_text TEXT NOT NULL,
    fact_type ENUM('user','technical','social','preference','project','financial') NOT NULL,
    confidence FLOAT DEFAULT 1.0,               -- 0.0-1.0 (extraccion auto = ~0.7)
    source_message_id INT NULL,
    source_facet VARCHAR(20),
    is_verified BOOLEAN DEFAULT FALSE,           -- revisado por Fernando?
    verified_at TIMESTAMP NULL,
    expires_at TIMESTAMP NULL,                   -- hechos temporales
    embedding VECTOR(768) NULL,                  -- busqueda semantica (Fase 2)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    user_id INT NULL,                            -- scope individual (memoria de dos niveles)
    project_id INT NULL,                         -- scope de proyecto (compartida)
    superseded_by INT NULL,                      -- si no-NULL: reemplazado por una correccion
    superseded_at TIMESTAMP NULL,
    FOREIGN KEY (source_message_id) REFERENCES messages(id) ON DELETE SET NULL,
    INDEX idx_fact_type (fact_type),
    INDEX idx_confidence (confidence),
    INDEX idx_expires (expires_at),
    INDEX idx_facts_user (user_id),
    INDEX idx_facts_project (project_id),
    INDEX idx_facts_active (superseded_by),
    FULLTEXT INDEX ft_fact_text (fact_text)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 4. DECISIONS — por que se eligio X. Auditoria del proyecto.
-- ------------------------------------------------------------
CREATE TABLE decisions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    decision_uuid CHAR(36) NOT NULL UNIQUE,
    title VARCHAR(255) NOT NULL,
    context TEXT,
    options JSON,
    chosen_option VARCHAR(255),
    reasoning TEXT NOT NULL,
    outcome TEXT NULL,
    project_id INT NULL,
    made_by_facet VARCHAR(20),
    made_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_project (project_id),
    INDEX idx_made_at (made_at)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 5. PROJECTS — contexto. Base tambien para el futuro financiero.
-- ------------------------------------------------------------
CREATE TABLE projects (
    id INT AUTO_INCREMENT PRIMARY KEY,
    project_uuid CHAR(36) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status ENUM('planning','active','paused','completed','archived') DEFAULT 'planning',
    start_date DATE,
    target_date DATE NULL,
    completion_date DATE NULL,
    github_repo VARCHAR(255) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    FULLTEXT INDEX ft_project_name (name)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 6. ERRORS — para debuggear JAX mismo (el "Napkin").
-- ------------------------------------------------------------
CREATE TABLE errors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    error_uuid CHAR(36) NOT NULL UNIQUE,
    facet VARCHAR(20) NOT NULL,
    error_type VARCHAR(100),
    error_message TEXT,
    user_input TEXT,
    conversation_id INT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE SET NULL,
    INDEX idx_facet (facet),
    INDEX idx_resolved (resolved),
    INDEX idx_created (created_at)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 7. PEOPLE — relaciones. honor_memory=TRUE para Jairo Urbina.
-- ------------------------------------------------------------
CREATE TABLE people (
    id INT AUTO_INCREMENT PRIMARY KEY,
    person_uuid CHAR(36) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    nickname VARCHAR(100) NULL,
    role VARCHAR(100) NULL,
    relationship_start DATE NULL,
    notes TEXT,
    honor_memory BOOLEAN DEFAULT FALSE,
    last_mentioned DATE NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_name (name),
    INDEX idx_honor_memory (honor_memory),
    FULLTEXT INDEX ft_notes (notes)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 8. ACTION_ITEMS — pendientes que JAX debe recordar.
-- ------------------------------------------------------------
CREATE TABLE action_items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    action_uuid CHAR(36) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    status ENUM('pending','done','cancelled','delegated') DEFAULT 'pending',
    due_date DATE NULL,
    reminder_date DATETIME NULL,                 -- para push de Telegram
    context_facet VARCHAR(20),
    source_conversation_id INT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (source_conversation_id) REFERENCES conversations(id) ON DELETE SET NULL,
    INDEX idx_status (status),
    INDEX idx_reminder (reminder_date),
    INDEX idx_due (due_date)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 9. JAX_METADATA — estado del sistema. ('key' es palabra
--    reservada, por eso va con backticks.)
-- ------------------------------------------------------------
CREATE TABLE jax_metadata (
    `key` VARCHAR(100) PRIMARY KEY,
    `value` TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Semilla de metadata e identidad
INSERT INTO jax_metadata (`key`, `value`) VALUES
    ('version', '2.0.0'),
    ('first_boot', NOW()),
    ('total_conversations', '0');

-- Jairo Urbina, la persona que JAX honra
INSERT INTO people (person_uuid, name, nickname, role, notes, honor_memory)
VALUES (
    UUID(),
    'Jairo Urbina',
    NULL,
    'amigo y socio',
    'Pionero del software libre en Honduras. JAX honra su memoria; su espiritu vive en el sistema. Su nombre aparece en el system prompt de cada faceta.',
    TRUE
);
