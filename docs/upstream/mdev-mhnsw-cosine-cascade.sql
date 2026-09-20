-- =====================================================================
-- MariaDB MHNSW vector index: rows become unreachable through the index
-- after a FOREIGN KEY ... ON DELETE CASCADE deletion.
--
-- Self-contained. No client, no extension, no large vectors: VECTOR(4)
-- and the built-in `seq_1_to_N` sequence engine are enough.
--
-- Measured on 12.3.3-MariaDB-ubu2404 (mariadb.org binary distribution).
--
-- EXPECTED: both SELECTs return 10 (the table holds 25 live rows).
-- ACTUAL:   the indexed path returns FEWER than the IGNORE INDEX path.
--           Observed 7/10 and 0/10 across runs; 8/8 runs reproduced with
--           M='16'. The control (direct DELETE, no cascade) was clean 8/8.
-- =====================================================================

DROP DATABASE IF EXISTS mhnsw_cascade_bug;
CREATE DATABASE mhnsw_cascade_bug;
USE mhnsw_cascade_bug;

CREATE TABLE parent (id INT PRIMARY KEY) ENGINE=InnoDB;

CREATE TABLE child (
  id        INT AUTO_INCREMENT PRIMARY KEY,
  parent_id INT NOT NULL,
  v         VECTOR(4) NOT NULL,
  VECTOR INDEX idx_v (v) `DISTANCE`='cosine' `M`='16',
  CONSTRAINT fk_child FOREIGN KEY (parent_id)
    REFERENCES parent (id) ON DELETE CASCADE
) ENGINE=InnoDB;

INSERT INTO parent VALUES (1), (2);

-- 300 rows under parent 1
INSERT INTO child (parent_id, v)
SELECT 1, VEC_FromText(CONCAT('[', seq/1000, ',', 1-seq/1000, ',0.1,0.2]'))
FROM seq_1_to_300;

-- The trigger: a CASCADE deletion.
-- Replace with `DELETE FROM child WHERE parent_id = 1;` for the control
-- (direct deletion) -- that one behaves correctly.
DELETE FROM parent WHERE id = 1;

-- 25 fresh rows under parent 2
INSERT INTO child (parent_id, v)
SELECT 2, VEC_FromText(CONCAT('[', seq/100, ',', 1-seq/100, ',0.3,0.4]'))
FROM seq_1_to_25;

SELECT COUNT(*) AS live_rows FROM child;                 -- 25

SELECT 'through the index' AS path, COUNT(*) AS returned FROM (
  SELECT id FROM child
  ORDER BY VEC_DISTANCE_COSINE(v, VEC_FromText('[0.5,0.5,0.3,0.4]')) LIMIT 10) t;

SELECT 'IGNORE INDEX'      AS path, COUNT(*) AS returned FROM (
  SELECT id FROM child IGNORE INDEX (idx_v)
  ORDER BY VEC_DISTANCE_COSINE(v, VEC_FromText('[0.5,0.5,0.3,0.4]')) LIMIT 10) t;

-- Rebuilding the index restores correct results, without touching data:
--   ALTER TABLE child DROP INDEX idx_v;
--   ALTER TABLE child ADD VECTOR INDEX idx_v (v) `DISTANCE`='cosine' `M`='16';
-- OPTIMIZE TABLE child;  -- does NOT help.

DROP DATABASE mhnsw_cascade_bug;
