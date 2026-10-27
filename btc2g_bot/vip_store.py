"""Transactional payment ledger shared by both payment methods."""
import hashlib
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from vip_product import ACCESS_DAYS


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY, user_id INTEGER NOT NULL,
                    provider TEXT NOT NULL, amount INTEGER NOT NULL, currency TEXT NOT NULL,
                    created INTEGER NOT NULL, token_hash TEXT UNIQUE,
                    charge TEXT, paid INTEGER, revoked INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(provider, charge));
                CREATE TABLE IF NOT EXISTS access (
                    user_id INTEGER PRIMARY KEY, until INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS refunds (
                    provider TEXT NOT NULL, charge TEXT NOT NULL,
                    PRIMARY KEY(provider, charge));
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, user_id, provider, amount, currency, now=None, with_token=False):
        if type(user_id) is not int or user_id <= 0 or type(amount) is not int or amount <= 0:
            raise ValueError('Invalid order')
        if (provider, currency) not in {('stripe', 'USD'), ('stars', 'XTR')}:
            raise ValueError('Invalid provider/currency')
        order_id = secrets.token_urlsafe(24)
        token = secrets.token_urlsafe(32) if with_token else None
        with self.connection() as db:
            db.execute('INSERT INTO orders(id,user_id,provider,amount,currency,created,token_hash) VALUES(?,?,?,?,?,?,?)',
                       (order_id, user_id, provider, amount, currency, int(time.time() if now is None else now),
                        hashlib.sha256(token.encode()).hexdigest() if token else None))
        return order_id, token

    def order(self, order_id):
        with self.connection() as db:
            row = db.execute('SELECT * FROM orders WHERE id=?', (order_id,)).fetchone()
            return dict(row) if row else None

    def from_token(self, token, now=None):
        if not isinstance(token, str) or not 30 <= len(token) <= 100:
            raise ValueError('Invalid code')
        now = int(time.time() if now is None else now)
        with self.connection() as db:
            row = db.execute('SELECT * FROM orders WHERE token_hash=?',
                             (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if not row or row['paid'] is not None or row['created'] + 86400 <= now:
            raise ValueError('Code expired or already used')
        return dict(row)

    def check(self, order_id, provider, user_id, amount, currency):
        row = self.order(order_id)
        if not row or (row['provider'], row['user_id'], row['amount'], row['currency']) != (provider, user_id, amount, currency):
            raise ValueError('Payment does not match order')
        return row

    def grant(self, order_id, provider, user_id, amount, currency, charge, now=None):
        if not isinstance(charge, str) or not charge:
            raise ValueError('Missing charge ID')
        now = int(time.time() if now is None else now)
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM refunds WHERE provider=? AND charge=?', (provider, charge)).fetchone():
                return False
            row = db.execute('SELECT * FROM orders WHERE id=?', (order_id,)).fetchone()
            if not row or (row['provider'], row['user_id'], row['amount'], row['currency']) != (provider, user_id, amount, currency):
                raise ValueError('Payment does not match order')
            if row['paid'] is not None:
                if row['charge'] != charge:
                    raise ValueError('Order already paid with a different charge')
                return False
            previous = db.execute('SELECT until FROM access WHERE user_id=?', (user_id,)).fetchone()
            until = max(now, previous['until'] if previous else 0) + ACCESS_DAYS * 86400
            db.execute('UPDATE orders SET charge=?,paid=? WHERE id=?', (charge, now, order_id))
            db.execute('INSERT INTO access VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET until=excluded.until', (user_id, until))
        return True

    def expires(self, user_id):
        with self.connection() as db:
            row = db.execute('SELECT until FROM access WHERE user_id=?', (user_id,)).fetchone()
            return row['until'] if row else 0

    def revoke(self, provider, charge, now=None):
        """Remove one paid period on a full refund, idempotently."""
        now = int(time.time() if now is None else now)
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT OR IGNORE INTO refunds VALUES(?,?)', (provider, charge))
            row = db.execute('SELECT * FROM orders WHERE provider=? AND charge=?', (provider, charge)).fetchone()
            if not row or row['revoked']:
                return False
            db.execute('UPDATE orders SET revoked=1 WHERE id=?', (row['id'],))
            db.execute('UPDATE access SET until=MAX(?,until-?) WHERE user_id=?', (now, ACCESS_DAYS*86400, row['user_id']))
        return True
