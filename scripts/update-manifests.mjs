#!/usr/bin/env node
// Writes the version, the download address and the SHA-256 of the Windows zip of a release into the Scoop and winget
// manifests of packaging/windows, so that they always describe the latest release. The release workflow runs it once
// the release is published and commits the result.
//
//   node scripts/update-manifests.mjs <version> <SHA256SUMS.txt> [packaging/windows folder]
//
// <version> is a plain x.y.z (a pre-release is not what a package manager should install). Every field is replaced
// exactly once or the script stops without writing anything.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO = 'L-at-nnes/OpenScissors';

/** `text` with the one match of `pattern` (a regex with the part to keep in the groups) replaced by `value`. */
function replaceOnce(text, pattern, value, what) {
  const found = [...text.matchAll(new RegExp(pattern.source, pattern.flags.includes('g') ? pattern.flags : pattern.flags + 'g'))];
  if (found.length !== 1) throw new Error(`${what}: expected 1 place to update, found ${found.length}`);
  // The arguments are the match, the groups, then the position and the whole text: keep the groups only.
  return text.replace(pattern, (...args) => {
    const [before = '', after = ''] = args.slice(1, -2);
    return `${before}${value}${after}`;
  });
}

/** The SHA-256 listed for `name` in a `sha256sum` output. */
export function hashOf(sums, name) {
  for (const line of sums.split(/\r?\n/)) {
    const m = line.match(/^([0-9a-fA-F]{64})\s+\*?(\S+)\s*$/);
    if (m && m[2] === name) return m[1].toLowerCase();
  }
  throw new Error(`${name} is not in the checksums file`);
}

/** Update the manifests of `dir` (the packaging/windows folder) for `version`, with the checksums text `sums`. */
export function update(version, sums, dir) {
  if (!/^\d+\.\d+\.\d+$/.test(version)) throw new Error(`"${version}" is not a plain x.y.z version`);
  const zip = `OpenScissors-${version}-portable-win-x64.zip`;
  const url = `https://github.com/${REPO}/releases/download/v${version}/${zip}`;
  const hash = hashOf(sums, zip);

  const scoopFile = path.join(dir, 'scoop', 'openscissors.json');
  let scoop = fs.readFileSync(scoopFile, 'utf8');
  scoop = replaceOnce(scoop, /("version":\s*")[^"]*(")/, version, 'Scoop version');
  // The address of the zip itself; the one of `autoupdate` keeps its $version placeholders.
  scoop = replaceOnce(scoop, /("url":\s*")https:\/\/github\.com\/[^"$]*\/releases\/download\/v\d[^"$]*(")/, url, 'Scoop url');
  scoop = replaceOnce(scoop, /("hash":\s*")[^"]*(")/, hash, 'Scoop hash');
  JSON.parse(scoop); // never leave a manifest that does not parse

  const wingetDir = path.join(dir, 'winget');
  const installerFile = path.join(wingetDir, 'L-at-nnes.OpenScissors.installer.yaml');
  const versioned = ['L-at-nnes.OpenScissors.yaml', 'L-at-nnes.OpenScissors.locale.en-US.yaml', 'L-at-nnes.OpenScissors.installer.yaml'];
  const out = new Map([[scoopFile, scoop]]);
  for (const name of versioned) {
    const file = path.join(wingetDir, name);
    let text = fs.readFileSync(file, 'utf8');
    text = replaceOnce(text, /^(PackageVersion:\s*).*$/m, version, `winget version in ${name}`);
    if (file === installerFile) {
      text = replaceOnce(text, /^(\s*InstallerUrl:\s*).*$/m, url, 'winget InstallerUrl');
      text = replaceOnce(text, /^(\s*InstallerSha256:\s*).*$/m, hash, 'winget InstallerSha256');
    }
    out.set(file, text);
  }
  // Everything was computed first: either all the files are written or none.
  for (const [file, text] of out) fs.writeFileSync(file, text);
  return { version, zip, hash, files: [...out.keys()] };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [version, sumsFile, dir] = process.argv.slice(2);
  if (!version || !sumsFile) {
    console.error('usage: node scripts/update-manifests.mjs <version> <SHA256SUMS.txt> [packaging/windows folder]');
    process.exit(2);
  }
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  try {
    const r = update(version, fs.readFileSync(sumsFile, 'utf8'), dir ?? path.join(root, 'packaging', 'windows'));
    console.log(`manifests updated for ${r.version}: ${r.zip} ${r.hash}`);
  } catch (e) {
    console.error(`update-manifests: ${e.message}`);
    process.exit(1);
  }
}
