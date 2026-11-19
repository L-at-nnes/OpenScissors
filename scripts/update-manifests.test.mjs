// node --test scripts/
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { hashOf, update } from './update-manifests.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const real = path.join(here, '..', 'packaging', 'windows');
const HASH = 'ab'.repeat(32);
const sums = (version) =>
  [
    `${'11'.repeat(32)}  OpenScissors-${version}-portable-darwin-arm64.zip`,
    `${'22'.repeat(32)}  OpenScissors-${version}-portable-linux-x64.zip`,
    `${HASH.toUpperCase()}  OpenScissors-${version}-portable-win-x64.zip`,
  ].join('\n') + '\n';

/** A copy of the manifests in a folder of its own. */
function copy() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'osc-manifests-'));
  fs.cpSync(real, dir, { recursive: true });
  return dir;
}
const read = (dir, ...p) => fs.readFileSync(path.join(dir, ...p), 'utf8');

test('the version, the address and the hash reach every manifest', () => {
  const dir = copy();
  update('9.8.7', sums('9.8.7'), dir);
  const scoop = JSON.parse(read(dir, 'scoop', 'openscissors.json'));
  assert.equal(scoop.version, '9.8.7');
  assert.equal(scoop.architecture['64bit'].url, 'https://github.com/L-at-nnes/OpenScissors/releases/download/v9.8.7/OpenScissors-9.8.7-portable-win-x64.zip');
  assert.equal(scoop.architecture['64bit'].hash, HASH, 'lower case, whatever the checksums file says');
  assert.equal(scoop.autoupdate.architecture['64bit'].url, 'https://github.com/L-at-nnes/OpenScissors/releases/download/v$version/OpenScissors-$version-portable-win-x64.zip', 'autoupdate keeps its placeholders');
  assert.equal(scoop.autoupdate.hash.url, '$baseurl/SHA256SUMS.txt');
  for (const f of ['L-at-nnes.OpenScissors.yaml', 'L-at-nnes.OpenScissors.locale.en-US.yaml', 'L-at-nnes.OpenScissors.installer.yaml']) {
    assert.match(read(dir, 'winget', f), /^PackageVersion: 9\.8\.7$/m, f);
  }
  const installer = read(dir, 'winget', 'L-at-nnes.OpenScissors.installer.yaml');
  assert.match(installer, /^ {4}InstallerUrl: https:\/\/github\.com\/L-at-nnes\/OpenScissors\/releases\/download\/v9\.8\.7\/OpenScissors-9\.8\.7-portable-win-x64\.zip$/m);
  assert.match(installer, new RegExp(`^ {4}InstallerSha256: ${HASH}$`, 'm'));
});

test('running it again changes nothing', () => {
  const dir = copy();
  update('9.8.7', sums('9.8.7'), dir);
  const once = ['scoop/openscissors.json', 'winget/L-at-nnes.OpenScissors.installer.yaml'].map((f) => read(dir, ...f.split('/')));
  update('9.8.7', sums('9.8.7'), dir);
  const twice = ['scoop/openscissors.json', 'winget/L-at-nnes.OpenScissors.installer.yaml'].map((f) => read(dir, ...f.split('/')));
  assert.deepEqual(twice, once);
});

test('a bad version or a missing zip writes nothing', () => {
  const dir = copy();
  const before = read(dir, 'scoop', 'openscissors.json') + read(dir, 'winget', 'L-at-nnes.OpenScissors.installer.yaml');
  assert.throws(() => update('9.8.7-rc1', sums('9.8.7-rc1'), dir), /plain x\.y\.z/);
  assert.throws(() => update('9.8.7', sums('1.0.0'), dir), /not in the checksums/);
  assert.throws(() => update('9.8.7', 'not a checksums file', dir), /not in the checksums/);
  assert.equal(read(dir, 'scoop', 'openscissors.json') + read(dir, 'winget', 'L-at-nnes.OpenScissors.installer.yaml'), before);
});

test('a manifest that lost a field is refused before anything is written', () => {
  const dir = copy();
  const installer = path.join(dir, 'winget', 'L-at-nnes.OpenScissors.installer.yaml');
  fs.writeFileSync(installer, read(dir, 'winget', 'L-at-nnes.OpenScissors.installer.yaml').replace(/^ {4}InstallerSha256:.*\n/m, ''));
  const scoopBefore = read(dir, 'scoop', 'openscissors.json');
  assert.throws(() => update('9.8.7', sums('9.8.7'), dir), /InstallerSha256/);
  assert.equal(read(dir, 'scoop', 'openscissors.json'), scoopBefore, 'the Scoop file is not half updated');
});

test('hashOf reads the sha256sum format, with or without the binary star', () => {
  assert.equal(hashOf(`${HASH} *a.zip\r\n${'cd'.repeat(32)}  b.zip`, 'a.zip'), HASH);
  assert.throws(() => hashOf(`${HASH}  a.zip`, 'b.zip'), /not in the checksums/);
});
