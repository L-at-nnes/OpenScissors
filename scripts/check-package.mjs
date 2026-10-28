// Checks the portable archive that scripts/package-portable.mjs --zip wrote, the way a person would meet it: unzip it
// somewhere else and run what is inside. A zip that builds but does not start (a missing library, a forgotten file) is
// caught here, before it is published.
//
//   node scripts/check-package.mjs
//
// Everywhere: the command line program runs and lists the models. On Linux, also: no shared library is missing, and
// the window starts on a virtual display (xvfb-run) and is still running after a few seconds.
import { execFileSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const win = process.platform === 'win32';
const dist = path.join(root, 'dist');
const zip = fs.readdirSync(dist).find((f) => f.endsWith('.zip'));
if (!zip) {
  console.error('No archive in dist/: run node scripts/package-portable.mjs --build --zip first');
  process.exit(1);
}

const where = fs.mkdtempSync(path.join(os.tmpdir(), 'osc-check-'));
if (win) execFileSync('powershell', ['-NoProfile', '-Command', `Expand-Archive -Path '${path.join(dist, zip)}' -DestinationPath '${where}'`], { stdio: 'inherit' });
else execFileSync('unzip', ['-q', path.join(dist, zip), '-d', where], { stdio: 'inherit' });
const folder = path.join(where, fs.readdirSync(where)[0]);
const app = path.join(folder, win ? 'OpenScissors.exe' : 'OpenScissors');
const cli = path.join(folder, win ? 'openscissors-cli.exe' : 'openscissors-cli');
const env = { ...process.env, OPENSCISSORS_HOME: folder };
let failed = false;
const fail = (msg) => {
  console.error(`FAIL: ${msg}`);
  failed = true;
};

for (const f of [app, cli, path.join(folder, 'README.txt'), path.join(folder, 'licenses', 'LICENSE')]) {
  if (!fs.existsSync(f)) fail(`${path.relative(folder, f)} is not in the archive`);
}

const list = spawnSync(cli, ['models'], { env, encoding: 'utf8' });
if (list.status !== 0 || !list.stdout.includes('isnet-general-use')) fail(`${path.basename(cli)} models did not list the models (exit ${list.status}): ${list.stderr}`);
else console.log('ok: the command line program runs');

if (process.platform === 'linux') {
  for (const bin of [app, cli]) {
    const missing = spawnSync('ldd', [bin], { encoding: 'utf8' }).stdout.split('\n').filter((l) => l.includes('not found'));
    if (missing.length) fail(`${path.basename(bin)} needs libraries that are not there:\n${missing.join('\n')}`);
  }
  console.log('ok: no shared library is missing');
  const png = path.join(root, 'tests', 'data', 'synthetic_cutout.png');
  // `timeout` ends the program after 10 s with status 124: that means it was still running, so it started.
  const run = spawnSync('xvfb-run', ['-a', '-s', '-screen 0 1400x900x24', 'timeout', '10', app, png], { env, encoding: 'utf8' });
  if (run.status !== 124) fail(`the window did not stay open (exit ${run.status}): ${run.stderr}`);
  else console.log('ok: the window starts and stays open');
}

fs.rmSync(where, { recursive: true, force: true });
process.exit(failed ? 1 : 0);
