# Distribution

How OpenScissors reaches people, besides building it from source. `packaging/` holds the files, the workflows build
them.

| Form | Built by | Where the application keeps its files |
|---|---|---|
| Portable zip (Windows, Linux, macOS) | `release.yml` on a tag (`scripts/package-portable.mjs --zip`) | its own folder |
| AppImage (Linux x86-64) | `package.yml`, called by `release.yml` once the release is published, or `packaging/linux/build-appimage.sh` | `~/.local/share/OpenScissors` |
| OpenScissors.app (macOS Apple Silicon) | `package.yml`, or `packaging/macos/build-app.sh` on a Mac | `~/Library/Application Support/OpenScissors` |
| Scoop (Windows) | `packaging/windows/scoop/openscissors.json` (kept up to date by `release.yml`), in a bucket | the Scoop folder of the app (`models` and `data` persist across updates) |
| winget (Windows) | `packaging/windows/winget/` (kept up to date by `release.yml`), sent to `microsoft/winget-pkgs` | the folder winget unpacks to |

The folder is chosen by `openscissors_core::paths::root`: `OPENSCISSORS_HOME` when set, the folder of the executable
when it can be written and is not inside an `.app`, the user's profile otherwise. An AppImage is read-only, and writing
into an `.app` would break its signature, so both use the profile.

The AppImage and `.app` scripts and `package.yml` have not run on a machine yet (they were written on Windows): check
the first artifacts by hand: "Run workflow" on `package.yml` builds both as artifacts without touching a release.

## Linux

`packaging/linux/build-appimage.sh` builds the release, lays out an AppDir (the program, the command line tool, the
desktop entry, the AppStream metadata, a 256 px icon) and packs it with appimagetool, downloaded when no path is given.
It writes `dist/OpenScissors-<version>-x86_64.AppImage`. It needs the same system libraries as the build, and Python
with Pillow for the icon.

`io.github.l_at_nnes.OpenScissors.desktop` and `.metainfo.xml` are the desktop entry and the AppStream description; a
distribution or a Flatpak manifest can use them as they are. Check the metadata with `appstreamcli validate`. A Flatpak
build is not provided: it needs the Rust dependencies vendored for an offline build (`flatpak-cargo-generator`).

To sign an AppImage, pass `--sign` (and `--sign-key <id>`) to appimagetool with a GPG key; AppImageUpdate and the
`--appimage-signature` option of the AppImage show the signature.

## macOS

`packaging/macos/build-app.sh` builds the release and the bundle (`Info.plist`, the `.icns` icon made with `sips` and
`iconutil`, the command line tool inside `Contents/MacOS`), signs it and zips it with `ditto` into
`dist/OpenScissors-<version>-macos-arm64.zip`. There is no Intel build: ONNX Runtime ships no x86-64 macOS binary.

Without an argument the bundle is signed ad hoc: Gatekeeper then asks for a right click > Open the first time. For a
bundle that opens with a double click on any Mac:

1. Sign with a Developer ID Application certificate (Apple Developer Program):
   `packaging/macos/build-app.sh "Developer ID Application: Name (TEAMID)"` (hardened runtime, timestamp).
2. Notarize: `xcrun notarytool submit dist/OpenScissors-<version>-macos-arm64.zip --keychain-profile <profile> --wait`
   (store the profile once with `xcrun notarytool store-credentials`).
3. Staple the ticket to the bundle and zip it again: `xcrun stapler staple dist/OpenScissors.app`, then
   `ditto -c -k --keepParent dist/OpenScissors.app dist/OpenScissors-<version>-macos-arm64.zip`.

In CI, import the certificate into a temporary keychain from a secret (for instance with
`apple-actions/import-codesign-certs`) before the script, and pass the notarization credentials as secrets.

## Windows

The portable zip needs no installer. Two package managers can install it:

- **Scoop**: put `packaging/windows/scoop/openscissors.json` in a bucket (a Git repository with a `bucket/` folder),
  and install with `scoop bucket add <name> <url>` then `scoop install openscissors`. `checkver` and `autoupdate` let
  `checkver.ps1 -u` follow the next releases (the hash is read from `SHA256SUMS.txt`).
- **winget**: check the three files with `winget validate packaging/windows/winget`, try them with
  `winget install --manifest packaging/windows/winget`, then propose them to `microsoft/winget-pkgs` under
  `manifests/l/L-at-nnes/OpenScissors/<version>/` (`wingetcreate update` does the version, URL and hash of the next
  releases).

Nothing is filled in by hand: once a release is published, the `manifests` job of `release.yml` runs
`scripts/update-manifests.mjs`, which writes the version, the download address and the SHA-256 of the Windows zip
(read from the `SHA256SUMS.txt` of the release) into these four files, and commits the result to `main`. A
pre-release tag (`v2.1.0-rc1`) leaves them alone. To do the same by hand:
`node scripts/update-manifests.mjs 2.0.0 SHA256SUMS.txt`.

**Signing.** An unsigned executable makes SmartScreen warn until it has gained reputation. To sign it, use an
Authenticode certificate (an OV or EV code signing certificate, or Azure Trusted Signing) with
`signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /f <cert.pfx> OpenScissors.exe openscissors-cli.exe`
(or `/dlib` and the Trusted Signing client) after the build and before the zip; in CI keep the certificate or the
Trusted Signing credentials in secrets. The SHA-256 of the zip changes with the signature; the `manifests` job reads
it from the release, so it follows by itself.
