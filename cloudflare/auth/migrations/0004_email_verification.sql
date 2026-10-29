ALTER TABLE users ADD COLUMN email TEXT;
ALTER TABLE users ADD COLUMN email_key TEXT;
ALTER TABLE users ADD COLUMN email_verified_at INTEGER;
ALTER TABLE users ADD COLUMN email_verified_day TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_key
  ON users(email_key) WHERE email_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS email_verification_tokens (
  token_hash TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  email_key TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_email_verification_user
  ON email_verification_tokens(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_email_verification_expiry
  ON email_verification_tokens(expires_at);
