const { spawnSync } = require('node:child_process');
const { readdirSync } = require('node:fs');

const compose = ['compose', '-f', 'compose.local.yml'];
const databaseUrl = 'postgresql://btc2g_local:btc2g_local_only@127.0.0.1:55432/btc2g_audience_local';

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { stdio: 'inherit', shell: false, ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} exited with status ${result.status}`);
}

try {
  run('docker', [...compose, 'up', '-d', '--wait']);
  const tests = readdirSync('tests').filter(name => name.endsWith('.test.cjs')).sort().map(name => `tests/${name}`);
  run(process.execPath, ['--test', ...tests], {
    env: { ...process.env, TEST_DATABASE_URL: databaseUrl }
  });
  run(process.execPath, ['migrate.cjs'], {
    env: { ...process.env, DATABASE_URL: databaseUrl }
  });
  console.log('Local PostgreSQL integration and migrations passed. No real email was sent.');
} finally {
  const stopped = spawnSync('docker', [...compose, 'down'], { stdio: 'inherit', shell: false });
  if (stopped.error || stopped.status !== 0) console.error('Local PostgreSQL cleanup needs attention.');
}
