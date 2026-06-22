#!/usr/bin/env bash
# Debian package (bundles the self-contained server binary + GUI). Installs on Debian,
# Kali, Whonix, Ubuntu and forks. CI-targeted.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist-pkg
[ -x dist-bin/crucible-serve ] || pyinstaller --onefile --name crucible-serve --paths backend \
  --collect-submodules crucible --exclude-module torch --exclude-module transformers \
  --exclude-module lm_eval --exclude-module datasets --collect-all uvicorn --collect-all fastapi \
  packaging/serve_main.py --distpath dist-bin
VER=$(grep -m1 '^version' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
PKG=build/crucible_${VER}_amd64
rm -rf "$PKG"; mkdir -p "$PKG/DEBIAN" "$PKG/usr/bin" "$PKG/usr/share/crucible"
cp dist-bin/crucible-serve "$PKG/usr/bin/"
cp -r frontend/dist "$PKG/usr/share/crucible/dist"
cat > "$PKG/usr/bin/crucible-gui" <<'LAUNCH'
#!/bin/bash
export CRUCIBLE_STATIC=/usr/share/crucible/dist
( sleep 2; xdg-open http://127.0.0.1:8400 >/dev/null 2>&1 || true ) &
exec /usr/bin/crucible-serve
LAUNCH
chmod +x "$PKG/usr/bin/crucible-gui"
cat > "$PKG/DEBIAN/control" <<CTRL
Package: crucible
Version: ${VER}
Section: utils
Priority: optional
Architecture: amd64
Maintainer: Crucible
Description: Local LLM censorship lab + agentic harness (backend + GUI)
CTRL
dpkg-deb --build "$PKG" "dist-pkg/crucible_${VER}_amd64.deb"
