BEGIN;
ALTER TABLE audience_subscriptions ADD COLUMN IF NOT EXISTS weekly_digest BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE audience_subscriptions ADD COLUMN IF NOT EXISTS digest_preferences_at TIMESTAMPTZ;
ALTER TABLE audience_subscriptions ADD COLUMN IF NOT EXISTS digest_consent_at TIMESTAMPTZ;
ALTER TABLE audience_subscriptions ADD COLUMN IF NOT EXISTS digest_unsubscribe_token TEXT UNIQUE;
ALTER TABLE audience_tokens ADD COLUMN IF NOT EXISTS weekly_digest BOOLEAN;
CREATE TABLE IF NOT EXISTS audience_digest_deliveries (
 email TEXT NOT NULL REFERENCES audience_subscriptions(email) ON DELETE CASCADE,
 issue_day DATE NOT NULL,
 state TEXT NOT NULL DEFAULT 'sending',
 provider_id TEXT,
 attempted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY(email,issue_day)
);
COMMIT;
