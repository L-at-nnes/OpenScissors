#!/usr/bin/env bash
# Builds OpenScissors.app and dist/OpenScissors-<version>-macos-arm64.zip from a release build (on a Mac, see
# docs/DISTRIBUTION.md).
#
#   packaging/macos/build-app.sh ["Developer ID Application: Name (TEAMID)"]
#
# Without an identity the bundle is signed ad hoc: it runs on the Mac that built it, and elsewhere after a right
# click > Open. With a Developer ID it is signed with the hardened runtime, ready for `xcrun notarytool`.
# Inside a bundle the application keeps its models and settings in ~/Library/Application Support/OpenScissors.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)"
identity="${1:--}"

cargo build --release -p openscissors -p openscissors-cli

app="dist/OpenScissors.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp target/release/openscissors "$app/Contents/MacOS/OpenScissors"
cp target/release/openscissors-cli "$app/Contents/MacOS/openscissors-cli"
cp LICENSE NOTICE THIRD_PARTY.md "$app/Contents/Resources/"

# The icon: every size iconutil expects, from the 128 px picture of the application.
iconset="$(mktemp -d)/OpenScissors.iconset"
mkdir -p "$iconset"
for size in 16 32 64 128 256 512; do
  sips -z "$size" "$size" crates/app/assets/icon.png --out "$iconset/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" crates/app/assets/icon.png --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$app/Contents/Resources/OpenScissors.icns"

cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>OpenScissors</string>
  <key>CFBundleDisplayName</key><string>OpenScissors</string>
  <key>CFBundleIdentifier</key><string>io.github.l-at-nnes.OpenScissors</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleExecutable</key><string>OpenScissors</string>
  <key>CFBundleIconFile</key><string>OpenScissors</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeRole</key><string>Editor</string>
      <key>LSItemContentTypes</key>
      <array>
        <string>public.png</string><string>public.jpeg</string><string>org.webmproject.webp</string>
        <string>public.tiff</string><string>com.microsoft.bmp</string><string>com.compuserve.gif</string>
      </array>
    </dict>
  </array>
</dict>
</plist>
EOF

if [ "$identity" = "-" ]; then
  codesign --force --deep --sign - "$app"
else
  codesign --force --deep --options runtime --timestamp --sign "$identity" "$app"
fi
codesign --verify --deep --strict "$app"

zip="dist/OpenScissors-$version-macos-arm64.zip"
rm -f "$zip"
ditto -c -k --keepParent "$app" "$zip"
echo "$zip"
