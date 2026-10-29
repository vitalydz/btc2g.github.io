ALTER TABLE daily_visitors ADD COLUMN acquisition_source TEXT;
ALTER TABLE users ADD COLUMN acquisition_source TEXT;

CREATE INDEX IF NOT EXISTS idx_daily_visitors_source_day
  ON daily_visitors(acquisition_source, day);
CREATE INDEX IF NOT EXISTS idx_users_source_day
  ON users(acquisition_source, created_day);
