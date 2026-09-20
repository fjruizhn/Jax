-- ============================================================
--  JAX 2.0 — Memoria Persistente
--  MariaDB 11.8+ (VECTOR nativo). Vive en hall9000.
--
--  ACTUALIZADO 2026-09-11 (segunda vez, mismo dia): `messages` gana
--  user_id/project_id --el scope desnormalizado desde conversations, para que
--  la busqueda no tenga que unir y pueda usar el indice vectorial-- mas el
--  indice `idx_msg_scope`, y `idx_embedding` se reconstruyo con `M`=16 (antes
--  el default 6): con M=6 el recall@5 tocaba techo en 94 %, con M=16 llega a
--  93,3 % ya con ef_search=400 en vez de 1000. Las columnas las aplica
--  jax/memory/migrations.py, que es el migrador real -- este archivo describe,
--  no aplica.
--
--  ACTUALIZADO 2026-09-12: corte de embeddings a bge-m3 (1024). `messages`
--  y `facts` ganan `embedding_bge_m3` y el VECTOR KEY se movio a ella
--  (MariaDB no admite dos por tabla). `embedding` (768, nomic) se RETIRO el
--  mismo dia (`scripts/migrar_embeddings.py retirar`, a pedido de Fernando).
--  Bloques regenerados desde
--  `SHOW CREATE TABLE` de produccion. Ver
--  docs/runbooks/migracion-embeddings-bge-m3.md.
--
--  REGENERADO 2026-09-11 desde `SHOW CREATE TABLE` de produccion
--  (jax_memory): la version anterior llevaba meses desactualizada --
--  le faltaban conversations.tenant_id/user_id/project_id,
--  decisions.user_id y action_items.user_id/project_id, el ENUM de
--  messages.role tenia 5 de 8 valores, y `embedding` figuraba NULL sin
--  default ni VECTOR KEY. Ningun ALTER del codigo explicaba la diferencia
--  (columnas agregadas a mano en produccion).
--
--  ESTE ARCHIVO ES LA FUENTE: se ejecuta en CI (tests/
--  test_memory_vector_zero_io.py crea sus tablas desde aca) y
--  scripts/check_memory_schema_drift.py lo compara contra la base viva.
--  Un ALTER a mano en produccion sin actualizar este archivo pone ese
--  chequeo en rojo.
--
--  embedding: VECTOR NOT NULL con DEFAULT vector cero porque el VECTOR KEY
--  exige NOT NULL. Una fila "sin embedding" queda en ceros, no en NULL --
--  ver jax/memory/db.py::_nonzero_embedding_sql y backfill_zero_embeddings.
--
--  En memoria de Jairo Urbina.
-- ============================================================

-- Base con utf8mb4 (emojis, modismos, acentos)
CREATE DATABASE IF NOT EXISTS jax_memory
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE jax_memory;

CREATE TABLE `conversations` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `conversation_uuid` char(36) NOT NULL,
  `started_at` timestamp NULL DEFAULT current_timestamp(),
  `ended_at` timestamp NULL DEFAULT NULL,
  `total_turns` int(11) DEFAULT 0,
  `source` varchar(20) DEFAULT 'terminal',
  `tenant_id` int(11) DEFAULT NULL,
  `user_id` int(11) DEFAULT NULL,
  `project_id` int(11) DEFAULT NULL,
  `memory_processed` tinyint(1) DEFAULT 0,
  `memory_processed_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `conversation_uuid` (`conversation_uuid`),
  KEY `idx_started` (`started_at`),
  KEY `idx_processed` (`memory_processed`),
  KEY `idx_conv_user` (`user_id`),
  KEY `idx_conv_project` (`project_id`),
  KEY `idx_conv_tenant` (`tenant_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `messages` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `conversation_id` int(11) NOT NULL,
  `turn_number` int(11) NOT NULL,
  `role` enum('user','jax_local','jekyll','hyde','hipatia','thot','kimi','ada') NOT NULL,
  `content` text NOT NULL,
  `facet_used` varchar(20) DEFAULT NULL,
  `model` varchar(50) DEFAULT NULL,
  `latency_ms` int(11) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `user_id` int(11) DEFAULT NULL,
  `project_id` int(11) DEFAULT NULL,
  `embedding_bge_m3` vector(1024) NOT NULL DEFAULT VEC_FromText('[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'),
  PRIMARY KEY (`id`),
  KEY `idx_conversation` (`conversation_id`),
  KEY `idx_role` (`role`),
  KEY `idx_created` (`created_at`),
  KEY `idx_msg_scope` (`project_id`,`user_id`),
  VECTOR KEY `idx_embedding_bge_m3` (`embedding_bge_m3`) `DISTANCE`='cosine' `M`='16',
  CONSTRAINT `messages_ibfk_1` FOREIGN KEY (`conversation_id`) REFERENCES `conversations` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `facts` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `fact_uuid` char(36) NOT NULL,
  `fact_text` text NOT NULL,
  `fact_type` enum('user','technical','social','preference','project','financial') NOT NULL,
  `confidence` float DEFAULT 1,
  `source_message_id` int(11) DEFAULT NULL,
  `source_facet` varchar(20) DEFAULT NULL,
  `is_verified` tinyint(1) DEFAULT 0,
  `verified_at` timestamp NULL DEFAULT NULL,
  `verified_by` int(11) DEFAULT NULL,
  `superseded_by_user` int(11) DEFAULT NULL,
  `expires_at` timestamp NULL DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `user_id` int(11) DEFAULT NULL,
  `project_id` int(11) DEFAULT NULL,
  `superseded_by` int(11) DEFAULT NULL,
  `superseded_at` timestamp NULL DEFAULT NULL,
  `source_fact_ids` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`source_fact_ids`)),
  `importance` tinyint(4) DEFAULT NULL,
  `embedding_bge_m3` vector(1024) NOT NULL DEFAULT VEC_FromText('[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'),
  PRIMARY KEY (`id`),
  UNIQUE KEY `fact_uuid` (`fact_uuid`),
  KEY `source_message_id` (`source_message_id`),
  KEY `idx_fact_type` (`fact_type`),
  KEY `idx_confidence` (`confidence`),
  KEY `idx_expires` (`expires_at`),
  KEY `idx_facts_user` (`user_id`),
  KEY `idx_facts_project` (`project_id`),
  KEY `idx_facts_active` (`superseded_by`),
  KEY `idx_facts_revision` (`is_verified`,`expires_at`,`created_at`),
  FULLTEXT KEY `ft_fact_text` (`fact_text`),
  VECTOR KEY `idx_embedding_bge_m3` (`embedding_bge_m3`) `DISTANCE`='cosine',
  CONSTRAINT `facts_ibfk_1` FOREIGN KEY (`source_message_id`) REFERENCES `messages` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `decisions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `decision_uuid` char(36) NOT NULL,
  `title` varchar(255) NOT NULL,
  `context` text DEFAULT NULL,
  `options` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`options`)),
  `chosen_option` varchar(255) DEFAULT NULL,
  `reasoning` text NOT NULL,
  `outcome` text DEFAULT NULL,
  `project_id` int(11) DEFAULT NULL,
  `made_by_facet` varchar(20) DEFAULT NULL,
  `made_at` timestamp NULL DEFAULT current_timestamp(),
  `user_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `decision_uuid` (`decision_uuid`),
  KEY `idx_project` (`project_id`),
  KEY `idx_made_at` (`made_at`),
  KEY `idx_dec_user` (`user_id`),
  KEY `idx_dec_project` (`project_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `projects` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `project_uuid` char(36) NOT NULL,
  `name` varchar(255) NOT NULL,
  `description` text DEFAULT NULL,
  `status` enum('planning','active','paused','completed','archived') DEFAULT 'planning',
  `start_date` date DEFAULT NULL,
  `target_date` date DEFAULT NULL,
  `completion_date` date DEFAULT NULL,
  `github_repo` varchar(255) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `project_uuid` (`project_uuid`),
  KEY `idx_status` (`status`),
  FULLTEXT KEY `ft_project_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `errors` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `error_uuid` char(36) NOT NULL,
  `facet` varchar(20) NOT NULL,
  `error_type` varchar(100) DEFAULT NULL,
  `error_message` text DEFAULT NULL,
  `user_input` text DEFAULT NULL,
  `conversation_id` int(11) DEFAULT NULL,
  `resolved` tinyint(1) DEFAULT 0,
  `resolution_notes` text DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `resolved_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `error_uuid` (`error_uuid`),
  KEY `conversation_id` (`conversation_id`),
  KEY `idx_facet` (`facet`),
  KEY `idx_resolved` (`resolved`),
  KEY `idx_created` (`created_at`),
  CONSTRAINT `errors_ibfk_1` FOREIGN KEY (`conversation_id`) REFERENCES `conversations` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `people` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `person_uuid` char(36) NOT NULL,
  `name` varchar(255) NOT NULL,
  `nickname` varchar(100) DEFAULT NULL,
  `role` varchar(100) DEFAULT NULL,
  `relationship_start` date DEFAULT NULL,
  `notes` text DEFAULT NULL,
  `honor_memory` tinyint(1) DEFAULT 0,
  `last_mentioned` date DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `person_uuid` (`person_uuid`),
  KEY `idx_name` (`name`),
  KEY `idx_honor_memory` (`honor_memory`),
  FULLTEXT KEY `ft_notes` (`notes`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `action_items` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `action_uuid` char(36) NOT NULL,
  `description` text NOT NULL,
  `status` enum('pending','done','cancelled','delegated') DEFAULT 'pending',
  `due_date` date DEFAULT NULL,
  `reminder_date` datetime DEFAULT NULL,
  `context_facet` varchar(20) DEFAULT NULL,
  `source_conversation_id` int(11) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `completed_at` timestamp NULL DEFAULT NULL,
  `user_id` int(11) DEFAULT NULL,
  `project_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `action_uuid` (`action_uuid`),
  KEY `source_conversation_id` (`source_conversation_id`),
  KEY `idx_status` (`status`),
  KEY `idx_reminder` (`reminder_date`),
  KEY `idx_due` (`due_date`),
  KEY `idx_act_user` (`user_id`),
  KEY `idx_act_project` (`project_id`),
  CONSTRAINT `action_items_ibfk_1` FOREIGN KEY (`source_conversation_id`) REFERENCES `conversations` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `jax_metadata` (
  `key` varchar(100) NOT NULL,
  `value` text NOT NULL,
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
