let pool;
module.exports = {
 async query(text, values) {
  if (!process.env.DATABASE_URL) throw new Error('Database unavailable');
  if (!pool) {
   const { Pool } = require('pg');
   pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 3, connectionTimeoutMillis: 5000, query_timeout: 8000, statement_timeout: 8000, idleTimeoutMillis: 10000 });
   pool.on('error', () => console.error('Audience database connection failed'));
  }
  return pool.query(text, values);
 }
};
