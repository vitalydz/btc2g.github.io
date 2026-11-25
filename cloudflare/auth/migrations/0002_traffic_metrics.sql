CREATE TABLE IF NOT EXISTS daily_visitors (
  day TEXT NOT NULL,
  visitor_hash TEXT NOT NULL,
  first_path TEXT NOT NULL,
  last_path TEXT NOT NULL,
  first_seen_at INTEGER NOT NULL,
  last_seen_at INTEGER NOT NULL,
  views INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (day, visitor_hash)
);

CREATE INDEX IF NOT EXISTS idx_daily_visitors_day ON daily_visitors(day);

CREATE TABLE IF NOT EXISTS daily_pageviews (
  day TEXT NOT NULL,
  page_path TEXT NOT NULL,
  views INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (day, page_path)
);

CREATE INDEX IF NOT EXISTS idx_daily_pageviews_day ON daily_pageviews(day);
