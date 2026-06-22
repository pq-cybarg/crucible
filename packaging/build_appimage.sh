#!/usr/bin/env bash
# Portable Linux AppImage (backend + GUI) — runs on EVERY distro (Kali/Whonix/Tails/
# Debian/Arch forks/anything) with no install. CI-targeted (Ubuntu runner).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist-pkg
# self-contained server binary (no torch — control plane + GUI)
pyinstaller --onefile --name crucible-serve --paths backend --collect-submodules crucible \
  --exclude-module torch --exclude-module transformers --exclude-module lm_eval --exclude-module datasets \
  --collect-all uvicorn --collect-all fastapi packaging/serve_main.py --distpath dist-bin
APPDIR=build/Crucible.AppDir
rm -rf "$APPDIR"; mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/crucible/frontend"
cp dist-bin/crucible-serve "$APPDIR/usr/bin/"
cp -r frontend/dist "$APPDIR/usr/share/crucible/frontend/dist"
cp frontend/public/icon.svg "$APPDIR/crucible.svg"
cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
export CRUCIBLE_STATIC="$HERE/usr/share/crucible/frontend/dist"
( sleep 2; xdg-open http://127.0.0.1:8400 >/dev/null 2>&1 || true ) &
exec "$HERE/usr/bin/crucible-serve"
RUN
chmod +x "$APPDIR/AppRun"
cat > "$APPDIR/crucible.desktop" <<'DESK'
[Desktop Entry]
Name=Crucible
Exec=crucible-serve
Icon=crucible
Type=Application
Categories=Development;
DESK
wget -q https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage -O /tmp/ait || true
chmod +x /tmp/ait 2>/dev/null || true
ARCH=x86_64 /tmp/ait "$APPDIR" dist-pkg/Crucible-x86_64.AppImage || echo "appimagetool unavailable; AppDir at $APPDIR"
