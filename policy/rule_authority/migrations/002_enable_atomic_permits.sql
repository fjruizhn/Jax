-- Enable PERMIT only after the adapter can atomically persist its RulePermit.
-- Migration 001 remains immutable; direct application writes still pass through
-- the append-only adapter contract and database constraints/triggers.
DROP TRIGGER IF EXISTS trg_rule_decisions_no_permit_until_step6;
