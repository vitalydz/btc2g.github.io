-- Preserve the atomic account limit for users while exempting administrators.
DROP TRIGGER IF EXISTS private_ideas_account_limit;
CREATE TRIGGER private_ideas_account_limit
BEFORE INSERT ON private_ideas
WHEN NOT EXISTS (SELECT 1 FROM users WHERE id = NEW.user_id AND role = 'admin')
 AND (SELECT COUNT(*) FROM private_ideas WHERE user_id = NEW.user_id) >= 3
BEGIN
  SELECT RAISE(ABORT, 'idea_limit_reached');
END;
