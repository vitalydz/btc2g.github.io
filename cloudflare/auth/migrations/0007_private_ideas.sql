CREATE TABLE IF NOT EXISTS private_ideas (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  audience TEXT NOT NULL,
  problem TEXT NOT NULL,
  solution TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'submitted'
    CHECK (status IN ('submitted', 'reviewing', 'accepted', 'declined')),
  moderation_note TEXT,
  idempotency_key TEXT NOT NULL,
  accepted_terms_version TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE (user_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_private_ideas_user
  ON private_ideas(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_private_ideas_queue
  ON private_ideas(status, created_at DESC);

CREATE TRIGGER IF NOT EXISTS private_ideas_account_limit
BEFORE INSERT ON private_ideas
WHEN (SELECT COUNT(*) FROM private_ideas WHERE user_id = NEW.user_id) >= 3
BEGIN
  SELECT RAISE(ABORT, 'idea_limit_reached');
END;

CREATE TABLE IF NOT EXISTS private_idea_attempts (
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  window_start INTEGER NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (user_id, window_start)
);

CREATE INDEX IF NOT EXISTS idx_private_idea_attempts_window
  ON private_idea_attempts(window_start);
