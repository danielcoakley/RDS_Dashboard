ALTER TABLE worker_heartbeats ADD COLUMN last_connector_attempted INTEGER NOT NULL DEFAULT 0;
ALTER TABLE worker_heartbeats ADD COLUMN last_connector_succeeded INTEGER NOT NULL DEFAULT 0;
ALTER TABLE worker_heartbeats ADD COLUMN last_connector_failed INTEGER NOT NULL DEFAULT 0;
ALTER TABLE worker_heartbeats ADD COLUMN last_connector_skipped INTEGER NOT NULL DEFAULT 0;
