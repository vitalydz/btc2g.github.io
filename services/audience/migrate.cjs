const { Pool } = require('pg');
const fs = require('node:fs');
(async () => {
 if (!process.env.DATABASE_URL) throw new Error('DATABASE_URL is required');
 const pool = new Pool({ connectionString: process.env.DATABASE_URL });
 try { for (const name of fs.readdirSync(__dirname + '/migrations').filter(n => n.endsWith('.sql')).sort()) await pool.query(fs.readFileSync(__dirname + '/migrations/' + name, 'utf8')); console.log('Audience migration applied'); }
 finally { await pool.end(); }
})().catch(() => { console.error('Migration failed; inspect database connectivity and schema permissions'); process.exitCode = 1; });
