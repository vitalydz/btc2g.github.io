CREATE TABLE IF NOT EXISTS login_rate_limits (
  scope_key TEXT PRIMARY KEY,
  window_start INTEGER NOT NULL,
  failure_count INTEGER NOT NULL DEFAULT 0,
  locked_until INTEGER NOT NULL DEFAULT 0,
  last_failed_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_login_rate_limits_last_failed
  ON login_rate_limits(last_failed_at);
