BEGIN;
ALTER TABLE audience_subscriptions ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'en';
ALTER TABLE audience_subscriptions ADD COLUMN IF NOT EXISTS delivery_state TEXT NOT NULL DEFAULT 'legacy';
ALTER TABLE audience_subscriptions ADD COLUMN IF NOT EXISTS last_attempt_hash TEXT;
CREATE TABLE IF NOT EXISTS audience_tokens (
 token_hash TEXT PRIMARY KEY,
 unsubscribe_hash TEXT UNIQUE NOT NULL,
 email TEXT NOT NULL REFERENCES audience_subscriptions(email) ON DELETE CASCADE,
 language TEXT NOT NULL CHECK(language IN ('en','ru','he')),
 expires_at TIMESTAMPTZ NOT NULL,
 used_at TIMESTAMPTZ,
 delivery_state TEXT NOT NULL DEFAULT 'pending',
 created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO audience_tokens(token_hash,unsubscribe_hash,email,language,expires_at)
 SELECT token_hash,unsubscribe_hash,email,'en',token_expires_at FROM audience_subscriptions
 WHERE token_hash IS NOT NULL AND token_expires_at IS NOT NULL ON CONFLICT DO NOTHING;
COMMIT;
