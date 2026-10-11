// Builds the portable version of OpenScissors: one folder, nothing to install.
//
//   node scripts/package-portable.mjs [outDir] [--build] [--zip] [--no-cli]
//
//   --build     run `cargo build --release` first
//   --zip       also write <outDir>-<version>-portable-<os>.zip next to the folder (no models, no user data)
//   --no-cli    leave the command line tool (openscissors-cli) out of the package
//
// Result (default dist/OpenScissors):
//   OpenScissors.exe   the application; it needs nothing else to run
//   openscissors-cli.exe  the command line version (cut, predict, batch, models)
//   DirectML.dll       only for the optional GPU mode (Windows)
//   README.txt
//   licenses/          LICENSE, NOTICE, THIRD_PARTY.md
//   models/ data/      what the person downloads and saves; created empty, never overwritten by this script
//
// The application keeps every file it writes (settings, models, temporary files) in this folder.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const win = process.platform === 'win32';
const args = process.argv.slice(2);
const flag = (name) => args.includes(`--${name}`);
const withCli = !flag('no-cli');
const out = path.resolve(args.find((a) => !a.startsWith('--')) ?? path.join(root, 'dist', 'OpenScissors'));
const exe = win ? 'openscissors.exe' : 'openscissors';
const cli = win ? 'openscissors-cli.exe' : 'openscissors-cli';

if (flag('build')) {
  const packages = ['-p', 'openscissors', ...(withCli ? ['-p', 'openscissors-cli'] : [])];
  execFileSync('cargo', ['build', '--release', ...packages], { cwd: root, stdio: 'inherit' });
}

const built = path.join(root, 'target', 'release', exe);
if (!fs.existsSync(built)) {
  console.error(`Missing ${built}. Build first: cargo build --release -p openscissors (or pass --build)`);
  process.exit(1);
}

// Replace the program files, keep the person's models/ and data/.
fs.mkdirSync(out, { recursive: true });
for (const name of ['OpenScissors.exe', 'OpenScissors', 'DirectML.dll', cli, 'README.txt', 'LICENSE', 'NOTICE', 'THIRD_PARTY.md']) {
  fs.rmSync(path.join(out, name), { force: true });
}
fs.rmSync(path.join(out, 'licenses'), { recursive: true, force: true });
fs.rmSync(path.join(out, 'temp'), { recursive: true, force: true });

fs.mkdirSync(path.join(out, 'models'), { recursive: true });
fs.mkdirSync(path.join(out, 'data'), { recursive: true });
fs.copyFileSync(built, path.join(out, win ? 'OpenScissors.exe' : 'OpenScissors'));

const dml = path.join(root, 'target', 'release', 'DirectML.dll');
if (win && fs.existsSync(dml) && fs.statSync(dml).size > 0) fs.copyFileSync(dml, path.join(out, 'DirectML.dll'));
if (withCli) {
  const from = path.join(root, 'target', 'release', cli);
  if (fs.existsSync(from)) fs.copyFileSync(from, path.join(out, cli));
}

fs.mkdirSync(path.join(out, 'licenses'), { recursive: true });
for (const f of ['LICENSE', 'NOTICE', 'THIRD_PARTY.md']) fs.copyFileSync(path.join(root, f), path.join(out, 'licenses', f));

fs.writeFileSync(
  path.join(out, 'README.txt'),
  [
    'OpenScissors (portable)',
    '',
    `Run ${win ? 'OpenScissors.exe' : 'OpenScissors'}. Everything lives in this folder: move it, copy it, put it on a USB stick.`,
    '',
    '  openscissors-cli.exe   the same engine from a terminal: openscissors-cli --help',
    '  models/    the AI models you download (top bar, model chip)',
    '  data/      your settings, window position and saved defaults',
    '  temp/      scratch files, cleaned up automatically (created when the app runs)',
    '  licenses/  license of OpenScissors and of the components it uses',
    '',
    'To start again from scratch, delete data/. To free space, delete models/.',
    '',
    'Nothing is loaded at start. The model you chose (otherwise the lightest one you downloaded)',
    'loads with the first image. The application keeps to 25% of the memory of the PC; Settings',
    'can raise that to 50%.',
    '',
    'Your own model: create models/<name>/ with an .onnx file named model.onnx and a model.json',
    '(see the project README).',
    '',
  ].join('\n'),
);
console.log('Portable folder ready:', out);

if (flag('zip')) {
  const version = JSON.parse(execFileSync('cargo', ['metadata', '--no-deps', '--format-version', '1'], { cwd: root, encoding: 'utf8', maxBuffer: 1 << 26 })).packages.find((p) => p.name === 'openscissors').version;
  const zip = path.join(path.dirname(out), `${path.basename(out)}-${version}-portable-${win ? 'win' : process.platform}-${process.arch}.zip`);
  // The archive holds the program only: a staged copy without models/ and data/ contents.
  const stage = fs.mkdtempSync(path.join(path.dirname(out), '.stage-'));
  const folder = path.join(stage, path.basename(out));
  fs.cpSync(out, folder, { recursive: true });
  for (const d of ['models', 'data', 'temp']) {
    fs.rmSync(path.join(folder, d), { recursive: true, force: true });
    if (d !== 'temp') fs.mkdirSync(path.join(folder, d));
  }
  fs.rmSync(zip, { force: true });
  if (win) execFileSync('powershell', ['-NoProfile', '-Command', `Compress-Archive -Path '${folder}' -DestinationPath '${zip}' -CompressionLevel Optimal`], { stdio: 'inherit' });
  else execFileSync('zip', ['-qr', zip, path.basename(out)], { cwd: stage, stdio: 'inherit' });
  fs.rmSync(stage, { recursive: true, force: true });
  console.log('Archive written:', zip);
}
