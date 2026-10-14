-- ─────────────────────────────────────────────────────────────────────
-- Row-level security: every territory manager logs in as their own
-- Postgres role, and Postgres itself filters rows by territory.
--
-- Why enforce this in the database and not in the prompt or the app?
--   * A prompt ("only show North data") can be ignored or jailbroken.
--   * An app-level WHERE clause can be forgotten in one code path.
--   * A session variable (SET app.territory = ...) can be overwritten by
--     the generated SQL itself -- set_config() is callable from a SELECT.
-- current_user is fixed at login, so no SQL the LLM writes can widen it.
--
-- Run as the table owner (postgres). Login roles are created by
-- scripts/load_rx.py because their names depend on the territories.
-- ─────────────────────────────────────────────────────────────────────

-- Group role that holds the privileges; login roles inherit from it.
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rx_reader') THEN
        CREATE ROLE rx_reader NOLOGIN;
    END IF;
END $$;

GRANT USAGE  ON SCHEMA rx TO rx_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA rx TO rx_reader;   -- SELECT only: read-only by construction

ALTER TABLE rx.prescribers      ENABLE ROW LEVEL SECURITY;
ALTER TABLE rx.prescriptions    ENABLE ROW LEVEL SECURITY;
ALTER TABLE rx.user_territories ENABLE ROW LEVEL SECURITY;

CREATE POLICY territory_isolation ON rx.prescribers FOR SELECT TO rx_reader
    USING (territory_id IN (SELECT territory_id FROM rx.user_territories
                            WHERE db_role = current_user));

CREATE POLICY territory_isolation ON rx.prescriptions FOR SELECT TO rx_reader
    USING (territory_id IN (SELECT territory_id FROM rx.user_territories
                            WHERE db_role = current_user));

CREATE POLICY own_mapping_only ON rx.user_territories FOR SELECT TO rx_reader
    USING (db_role = current_user);

-- territories and drugs are reference data: visible to everyone, no policy needed.
