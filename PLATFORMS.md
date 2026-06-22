# Crucible — Platform Support & Build Matrix

All artifacts autobuild in `.github/workflows/build.yml`. Every build step was verified
locally before commit (PWA build, PyInstaller CLI binary, multi-arch Docker image + run).

## What runs where

| Platform | Artifact | Backend (control plane + GUI) | Abliteration adapter (torch) |
|---|---|---|---|
| **Windows 10** (11 intentionally NOT targeted) | CLI `.exe` + Docker Desktop | ✅ | optional (`pip install torch`) |
| **macOS** (Intel + Apple Silicon) | CLI binary + Docker | ✅ | ✅ (MPS) |
| **Linux x86_64** (any distro) | AppImage · `.deb` · Docker · CLI | ✅ | ✅ (CUDA/CPU) |
| **Linux ARM64 / Raspberry Pi** | Docker `arm64` · CLI arm64 | ✅ | CPU only / use a GPU node |
| **Kali · Whonix · Tails · forks** | AppImage · `.deb` · Docker | ✅ | optional |
| **Android** | Capacitor APK | thin client → remote node | n/a |
| **iOS** | Capacitor archive | thin client → remote node | n/a |
| **Any browser / OS** | installable PWA | thin client → remote node | n/a |

## Two tiers, by design

- **Full node** (desktop/server): runs the backend + GUI locally. Torch adapter is
  **opt-in** — the control plane (registry, guardrails, agent, evals, VCS, serving) is
  numpy-only and runs on a Raspberry Pi; the heavy abliteration/diagnosis adapter needs
  `torch`+`transformers` and a capable machine (or point at a GPU/Windows node).
- **Thin client** (mobile/PWA): the installable app/PWA points at a remote Crucible node
  (set the URL in the GUI's **node** field or `crucible --endpoint`). The 1.5 TB model
  lives on the server; the phone just drives it.

## "All Linux distros" — by format, not allowlist

There is **no verifiable public list of distros that have "promised to refuse age
verification"** — age verification is an app/website/jurisdiction-level concern, not
something baked into a Linux distro. So Crucible ships in **universal portable formats**:

- **AppImage** — single file, no install, runs on every distro (incl. amnesic Tails from USB).
- **Flatpak/`.deb`** — Debian/Kali/Whonix/Ubuntu and forks.
- **Multi-arch Docker** (`linux/amd64` + `linux/arm64`) — runs identically on any distro and on Raspberry Pi.

This covers **every** Linux edition — current, privacy-focused, or a fork released
tomorrow — without tracking anyone's policy stance.

## Build it

```bash
# locally
docker build -t crucible . && docker run -p 8400:8400 crucible    # backend + GUI, any OS w/ Docker
bash packaging/build_appimage.sh                                  # portable Linux (CI/Ubuntu)
pyinstaller --onefile --name crucible --paths backend packaging/cli_main.py   # native CLI

# in CI: push to main or tag v* → all platforms build & attach to the release
```
