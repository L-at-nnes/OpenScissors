#!/usr/bin/env bash
# Builds OpenScissors-<version>-x86_64.AppImage from a release build (see docs/DISTRIBUTION.md).
#
#   packaging/linux/build-appimage.sh [path/to/appimagetool]
#
# Needs: cargo, python3 with Pillow (to enlarge the icon), and appimagetool (downloaded when not given).
# An AppImage is read-only: the application keeps its models and settings in ~/.local/share/OpenScissors.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)"
id="io.github.l_at_nnes.OpenScissors"
tool="${1:-}"

cargo build --release -p openscissors -p openscissors-cli

appdir="$(mktemp -d)/OpenScissors.AppDir"
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/applications" "$appdir/usr/share/metainfo" \
  "$appdir/usr/share/icons/hicolor/256x256/apps" "$appdir/usr/share/doc/openscissors"
cp target/release/openscissors "$appdir/usr/bin/OpenScissors"
cp target/release/openscissors-cli "$appdir/usr/bin/openscissors-cli"
cp LICENSE NOTICE THIRD_PARTY.md "$appdir/usr/share/doc/openscissors/"
cp "packaging/linux/$id.desktop" "$appdir/usr/share/applications/"
cp "packaging/linux/$id.metainfo.xml" "$appdir/usr/share/metainfo/"
python3 -c "from PIL import Image; Image.open('crates/app/assets/icon.png').resize((256, 256), Image.LANCZOS).save('$appdir/usr/share/icons/hicolor/256x256/apps/$id.png')"
# appimagetool wants the desktop file and the icon at the top too.
cp "packaging/linux/$id.desktop" "$appdir/"
cp "$appdir/usr/share/icons/hicolor/256x256/apps/$id.png" "$appdir/$id.png"
cat > "$appdir/AppRun" <<'EOF'
#!/bin/sh
here="$(dirname "$(readlink -f "$0")")"
exec "$here/usr/bin/OpenScissors" "$@"
EOF
chmod +x "$appdir/AppRun"

if [ -z "$tool" ]; then
  tool="$(mktemp -d)/appimagetool"
  curl -fsSL -o "$tool" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
  chmod +x "$tool"
fi
mkdir -p dist
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$tool" "$appdir" "dist/OpenScissors-$version-x86_64.AppImage"
echo "dist/OpenScissors-$version-x86_64.AppImage"
