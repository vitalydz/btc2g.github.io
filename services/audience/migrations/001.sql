BEGIN;
CREATE TABLE IF NOT EXISTS audience_subscriptions (
 email TEXT PRIMARY KEY,
 consent_version TEXT NOT NULL,
 requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 token_hash TEXT UNIQUE,
 token_expires_at TIMESTAMPTZ,
 unsubscribe_hash TEXT UNIQUE NOT NULL,
 confirmed_at TIMESTAMPTZ,
 unsubscribed_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS audience_visitors (
 day DATE NOT NULL,
 visitor_hash TEXT NOT NULL,
 first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 views BIGINT NOT NULL DEFAULT 1 CHECK (views > 0),
 PRIMARY KEY(day, visitor_hash)
);
CREATE TABLE IF NOT EXISTS audience_rate_limits (
 bucket TEXT PRIMARY KEY,
 attempts INTEGER NOT NULL DEFAULT 1,
 expires_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS audience_meta (id INTEGER PRIMARY KEY CHECK(id=1), initialized_at TIMESTAMPTZ NOT NULL DEFAULT now());
INSERT INTO audience_meta(id) VALUES(1) ON CONFLICT DO NOTHING;
COMMIT;
